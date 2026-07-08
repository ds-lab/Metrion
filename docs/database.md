CREATE TABLE entity (
    id          TEXT PRIMARY KEY,
    parent_id   TEXT NULL,
    entity_type TEXT
);

CREATE TABLE cpu (
    id                 TEXT PRIMARY KEY REFERENCES entity (id),
    manufacturer       TEXT,
    model              TEXT,
    physical_cores     BIGINT,
    logical_cores      BIGINT,
    package_id         BIGINT,
    cache              BIGINT,
    hyperthreading     BOOLEAN,
    core_ids           TEXT,              -- CommaSeparatedIntegers TypeDecorator
    idle_power_package DOUBLE PRECISION,
    max_power_package  DOUBLE PRECISION
);

CREATE TABLE core (
    id              TEXT PRIMARY KEY REFERENCES entity (id),
    core_id         BIGINT NOT NULL,
    package_id      BIGINT NOT NULL,
    thread_siblings TEXT NOT NULL,       -- CommaSeparatedIntegers
    mhz             BIGINT NOT NULL
);

CREATE TABLE dram (
    id           TEXT PRIMARY KEY REFERENCES entity (id),
    package_id   BIGINT NOT NULL,
    idle_power   DOUBLE PRECISION,
    max_power    DOUBLE PRECISION,
    total_memory DOUBLE PRECISION
);

CREATE TABLE gpu (
    id                       TEXT PRIMARY KEY REFERENCES entity (id),
    manufacturer             TEXT,
    model                    TEXT,
    serial_number            TEXT,
    uuid                     TEXT,
    pci_id                   TEXT,
    idle_power               DOUBLE PRECISION,
    max_power                DOUBLE PRECISION,
    compute_capability_major INTEGER,
    compute_capability_minor INTEGER
);

CREATE TABLE host (
    id         TEXT PRIMARY KEY REFERENCES entity (id),
    name       TEXT,
    host_type  host_type_enum,
    OS         TEXT
);


CREATE TABLE network_interface (
    id          TEXT PRIMARY KEY REFERENCES entity (id),
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,       -- e.g. 'ETHERNET', 'INFINIBAND'
    mac_address TEXT
);

CREATE TABLE network_addresses (
    id             TEXT PRIMARY KEY,
    interface_id   TEXT NOT NULL REFERENCES network_interface (id),
    address        TEXT NOT NULL,
    family         TEXT NOT NULL,
    prefix_length  BIGINT NOT NULL
);


CREATE TABLE logical_entity (
    id                      TEXT PRIMARY KEY REFERENCES entity (id),
    native_thread_id        BIGINT,
    
    native_pid              BIGINT,
    pid_id                  TEXT,

    native_parent_tid       BIGINT,
    parent_tid_id           TEXT,
    
    native_parent_pid       BIGINT,
    parent_pid_id           TEXT,

    executable_name         TEXT,
    cmdline                 TEXT,

    start_time              TIMESTAMP,
    pid_start_time          TIMESTAMP,
    parent_tid_start_time   TIMESTAMP,
    parent_pid_start_time   TIMESTAMP,

    resource_group_id       BIGINT,
    resource_group_name     TEXT,
    user_id                 BIGINT,
    group_id                BIGINT
);

CREATE TABLE metric (
    name        TEXT PRIMARY KEY,
    unit        TEXT,
    description TEXT
);

CREATE TABLE measurement (
    id                  BIGINT PRIMARY KEY,
    value               DOUBLE PRECISION,
    metric_id           TEXT NOT NULL REFERENCES metric (name),
    physical_entity_id  TEXT NOT NULL REFERENCES entity (id),
    logical_entity_id   TEXT NULL,
    start_ts            TIMESTAMP,
    end_ts              TIMESTAMP
);

CREATE TABLE communication_endpoint (
    id         TEXT PRIMARY KEY,
    ip_address TEXT NOT NULL,
    port       BIGINT NOT NULL,
    transport  TEXT NOT NULL        -- e.g. 'TCP', 'UDP'
);

CREATE TABLE communication_measurement (
    measurement_id BIGINT PRIMARY KEY REFERENCES measurement (id),
    src_endpoint   TEXT NOT NULL,
    dst_endpoint   TEXT NOT NULL
);


CREATE TABLE metadata (
    id             TEXT PRIMARY KEY,
    reference_id   TEXT NOT NULL,
    metadata_type  TEXT NOT NULL,
    metadata_value TEXT NOT NULL
);
