use libbpf_rs::{AsRawLibbpf, Link, MapCore, MapFlags, Object, ObjectBuilder, RingBuffer, RingBufferBuilder, libbpf_sys};
use pyo3::exceptions::{PyKeyError, PyOSError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::os::fd::{AsFd, AsRawFd};

// The BPF objects compiled by build.rs, embedded so that nothing is compiled on the target.
const OBJECTS: &[(&str, &[u8])] = &[
    (
        "logical_entity_registry",
        include_bytes!(concat!(env!("OUT_DIR"), "/logical_entity_registry.bpf.o")),
    ),
    (
        "sched_switch",
        include_bytes!(concat!(env!("OUT_DIR"), "/sched_switch.bpf.o")),
    ),
    (
        "tcp_monitor",
        include_bytes!(concat!(env!("OUT_DIR"), "/tcp_monitor.bpf.o")),
    ),
];

fn os_error(err: libbpf_rs::Error) -> PyErr {
    PyOSError::new_err(format!("{err:#}"))
}

/// A loaded and attached BPF object.
///
/// Parameters
/// ----------
/// name : str
///     One of the embedded objects, e.g. ``"sched_switch"``.
/// max_entries : dict[str, int]
///     Map sizes to set before loading, by map name. A ring buffer's size is in bytes.
#[pyclass(unsendable)]
pub struct BpfProgram {
    // Fields drop in this order: the ring buffer before the links, the links before the object.
    ring_buffer: Option<RingBuffer<'static>>,
    _links: Vec<Link>,
    object: Object,
}

impl BpfProgram {
    fn map(&self, name: &str) -> PyResult<libbpf_rs::Map<'_>> {
        self.object
            .maps()
            .find(|map| map.name() == name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }
}

#[pymethods]
impl BpfProgram {
    #[new]
    #[pyo3(signature = (name, max_entries = HashMap::new()))]
    fn new(name: &str, mut max_entries: HashMap<String, u32>) -> PyResult<Self> {
        let (_, bytes) = OBJECTS
            .iter()
            .find(|(object_name, _)| *object_name == name)
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))?;

        let mut open = ObjectBuilder::default().open_memory(bytes).map_err(os_error)?;
        for mut map in open.maps_mut() {
            let map_name = map.name().to_string_lossy().into_owned();
            if let Some(count) = max_entries.remove(&map_name) {
                map.set_max_entries(count).map_err(os_error)?;
            }
        }
        // Whatever is left names no map of this object, most likely a typo.
        if let Some(unknown) = max_entries.keys().next() {
            return Err(PyKeyError::new_err(format!("{name} has no map {unknown}")));
        }
        let object = open.load().map_err(os_error)?;

        let mut links = Vec::new();
        for prog in object.progs_mut() {
            links.push(prog.attach().map_err(os_error)?);
        }

        Ok(Self {
            ring_buffer: None,
            _links: links,
            object,
        })
    }

    fn map_fd(&self, name: &str) -> PyResult<i32> {
        Ok(self.map(name)?.as_fd().as_raw_fd())
    }

    /// Store a perf event fd in a perf event array, as BCC's ``table[Key(index)] = Leaf(fd)``.
    fn set_perf_fd(&self, name: &str, index: u32, fd: i32) -> PyResult<()> {
        self.map(name)?
            .update(&index.to_ne_bytes(), &fd.to_ne_bytes(), MapFlags::ANY)
            .map_err(os_error)
    }

    /// Sum of a u64 per-CPU array entry over all CPUs.
    fn percpu_sum(&self, name: &str, key: u32) -> PyResult<u64> {
        let values = self
            .map(name)?
            .lookup_percpu(&key.to_ne_bytes(), MapFlags::ANY)
            .map_err(os_error)?
            .unwrap_or_default();
        Ok(values
            .iter()
            .map(|value| u64::from_ne_bytes(value[..8].try_into().unwrap()))
            .sum())
    }

    /// Call ``callback(data: bytes)`` for every record of the ring buffer map ``name``.
    ///
    /// A negative return value ends the current ``ring_buffer_consume`` pass. As with BCC's
    /// ctypes callbacks, an exception is reported as unraisable and the record counts as handled.
    fn open_ring_buffer(&mut self, name: &str, callback: Py<PyAny>) -> PyResult<()> {
        let map = self.map(name)?;
        let mut builder = RingBufferBuilder::new();
        builder
            .add(&map, move |data: &[u8]| {
                Python::attach(|py| match callback.call1(py, (PyBytes::new(py, data),)) {
                    Ok(ret) => ret.extract::<i32>(py).unwrap_or(0),
                    Err(err) => {
                        err.write_unraisable(py, Some(callback.bind(py)));
                        0
                    }
                })
            })
            .map_err(os_error)?;
        self.ring_buffer = Some(builder.build().map_err(os_error)?);
        Ok(())
    }

    /// Read every record available now, as BCC's ``ring_buffer_consume``.
    fn ring_buffer_consume(&self) -> PyResult<i32> {
        let ring_buffer = self
            .ring_buffer
            .as_ref()
            .ok_or_else(|| PyOSError::new_err("No ring buffers to poll"))?;
        // The raw call returns the callback's negative value instead of turning it into an error.
        Ok(unsafe { libbpf_sys::ring_buffer__consume(ring_buffer.as_libbpf_object().as_ptr()) })
    }
}
