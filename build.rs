use libbpf_cargo::SkeletonBuilder;
use std::env;
use std::path::PathBuf;

// (object name, source) of every BPF program; src/bpf.rs embeds the objects by name.
const PROGRAMS: &[(&str, &str)] = &[
    ("logical_entity_registry", "metrion/entity/logical_entity_registry.bpf.c"),
    ("sched_switch", "metrion/monitoring/data_source/cpu/sched_switch.bpf.c"),
    ("tcp_monitor", "metrion/monitoring/data_source/network/tcp_monitor.bpf.c"),
];

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for (name, source) in PROGRAMS {
        SkeletonBuilder::new()
            .source(source)
            .obj(out_dir.join(format!("{name}.bpf.o")))
            .clang_args(["-Isrc/bpf", "-Imetrion/monitoring/data_source/helpers"])
            .build()
            .unwrap_or_else(|err| panic!("failed to compile {source}: {err:#}"));
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed=src/bpf/vmlinux.h");
    println!("cargo:rerun-if-changed=metrion/monitoring/data_source/helpers/common_bpf.h");
}
