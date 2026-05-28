-- GO graph core schema.
-- Active ontology: GO-25 + GO extension terms.
-- Purpose: boring durable STORAGE surface for located -> staged -> approved graph items.
-- Runtime/control STATE belongs in the separate state Postgres surface, not here.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS lucidota_go;

CREATE TABLE IF NOT EXISTS lucidota_go.term_registry (
    term text PRIMARY KEY,
    term_number integer,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','archived','experimental')),
    definition text NOT NULL DEFAULT '',
    parent_term text,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_go.soul_registry (
    suuid integer PRIMARY KEY CHECK (suuid BETWEEN 1 AND 5000),
    soul_kind text NOT NULL CHECK (soul_kind IN ('KERNEL','VILLAGER_SOUL')),
    label text NOT NULL DEFAULT '',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','inactive','archived','lost','collapsed')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_layer (
    layer text PRIMARY KEY,
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','archived','experimental')),
    description text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_item (
    uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    term text NOT NULL REFERENCES lucidota_go.term_registry(term),
    label text NOT NULL DEFAULT '',
    status text NOT NULL CHECK (status IN (
        'located',
        'staged',
        'approved',
        'rejected',
        'superseded',
        'archived',
        'error_corrected',
        'lost',
        'collapsed'
    )),
    canonical_uuid uuid,
    time_on_graph timestamptz NOT NULL DEFAULT now(),
    location_at_on_graph text NOT NULL,
    location_real_at_added jsonb NOT NULL DEFAULT '{}'::jsonb,
    time_approved timestamptz,
    location_real_at_approved jsonb NOT NULL DEFAULT '{}'::jsonb,
    approval_scope text CHECK (
        approval_scope IS NULL OR approval_scope IN (
            'contemporaneous',
            'current',
            'historical',
            'unknown_current',
            'superseded'
        )
    ),
    operator_uuid uuid,
    ternary_valency integer NOT NULL DEFAULT 0 CHECK (ternary_valency IN (1, 0, -1)),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (status <> 'approved')
        OR (time_approved IS NOT NULL AND approval_scope IS NOT NULL AND operator_uuid IS NOT NULL)
    ),
    CHECK (
        (status <> 'approved' OR term <> 'CLAIM')
        OR nullif(btrim(coalesce(payload->>'evidence_note', payload->>'proof_note', '')), '') IS NOT NULL
    ),
    CHECK (
        (term <> 'HYPOTHESIS' OR coalesce(payload->>'inferred', 'false') <> 'true')
        OR (payload->>'confidence_bps') IN ('2','4','6','10','50','69','150')
    )
);

CREATE INDEX IF NOT EXISTS graph_item_term_status_idx
    ON lucidota_go.graph_item(term, status, created_at DESC);

CREATE INDEX IF NOT EXISTS graph_item_canonical_idx
    ON lucidota_go.graph_item(canonical_uuid);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_item_layer (
    uuid uuid NOT NULL REFERENCES lucidota_go.graph_item(uuid) ON DELETE CASCADE,
    layer text NOT NULL REFERENCES lucidota_go.graph_layer(layer),
    role text NOT NULL DEFAULT '',
    confidence_bps integer NOT NULL DEFAULT 0 CHECK (confidence_bps IN (0,2,4,6,10,50,69,150)),
    operator_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (uuid, layer, role)
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_edge (
    edge_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_uuid uuid NOT NULL REFERENCES lucidota_go.graph_item(uuid),
    target_uuid uuid NOT NULL REFERENCES lucidota_go.graph_item(uuid),
    edge_type text NOT NULL,
    term text REFERENCES lucidota_go.term_registry(term),
    relationship_family text CHECK (
        relationship_family IS NULL OR relationship_family IN (
            'intimate',
            'interpersonal',
            'vector',
            'possessive',
            'family',
            'lover',
            'enemy',
            'works'
        )
    ),
    status text NOT NULL DEFAULT 'staged' CHECK (status IN (
        'located',
        'staged',
        'approved',
        'rejected',
        'superseded',
        'archived',
        'error_corrected',
        'lost',
        'collapsed'
    )),
    valid_from timestamptz,
    valid_to timestamptz,
    current_status text NOT NULL DEFAULT 'unknown' CHECK (current_status IN ('yes','no','unknown')),
    current_unknown boolean NOT NULL DEFAULT true,
    will_be text NOT NULL DEFAULT '',
    location_vector text CHECK (
        location_vector IS NULL OR location_vector IN ('WAS_WITH','WERE_AT','ARE_IN')
    ),
    location_uuid uuid REFERENCES lucidota_go.graph_item(uuid),
    evidence_uuid uuid REFERENCES lucidota_go.graph_item(uuid),
    operator_uuid uuid,
    ternary_valency integer NOT NULL DEFAULT 0 CHECK (ternary_valency IN (1, 0, -1)),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS graph_edge_source_idx
    ON lucidota_go.graph_edge(source_uuid, edge_type, created_at DESC);

CREATE INDEX IF NOT EXISTS graph_edge_target_idx
    ON lucidota_go.graph_edge(target_uuid, edge_type, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_edge_layer (
    edge_uuid uuid NOT NULL REFERENCES lucidota_go.graph_edge(edge_uuid) ON DELETE CASCADE,
    layer text NOT NULL REFERENCES lucidota_go.graph_layer(layer),
    role text NOT NULL DEFAULT '',
    confidence_bps integer NOT NULL DEFAULT 0 CHECK (confidence_bps IN (0,2,4,6,10,50,69,150)),
    operator_uuid uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (edge_uuid, layer, role)
);

CREATE TABLE IF NOT EXISTS lucidota_go.graph_journal (
    journal_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_uuid uuid REFERENCES lucidota_go.graph_item(uuid),
    edge_uuid uuid REFERENCES lucidota_go.graph_edge(edge_uuid),
    operator_uuid uuid,
    action text NOT NULL CHECK (action IN (
        'locate',
        'stage',
        'approve',
        'reject',
        'merge',
        'split',
        'nest',
        'role_convert',
        'migrate',
        'correct_error',
        'archive',
        'mark_lost',
        'mark_collapsed',
        'comment'
    )),
    reason text NOT NULL DEFAULT '',
    before_state jsonb NOT NULL DEFAULT '{}'::jsonb,
    after_state jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS graph_journal_item_idx
    ON lucidota_go.graph_journal(item_uuid, created_at DESC);

CREATE INDEX IF NOT EXISTS graph_journal_edge_idx
    ON lucidota_go.graph_journal(edge_uuid, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_go.staging_packet (
    packet_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id text NOT NULL DEFAULT '',
    parser_name text NOT NULL DEFAULT 'go_fast_parser',
    proposed_term text REFERENCES lucidota_go.term_registry(term),
    raw_anchor text NOT NULL DEFAULT '',
    claim text NOT NULL DEFAULT '',
    proposed_item jsonb NOT NULL DEFAULT '{}'::jsonb,
    proposed_edges jsonb NOT NULL DEFAULT '[]'::jsonb,
    status text NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending',
        'approved',
        'rejected',
        'needs_repair',
        'comment'
    )),
    confidence_bps integer NOT NULL DEFAULT 0 CHECK (confidence_bps IN (0,2,4,6,10,50,69,150)),
    operator_uuid uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    decided_at timestamptz
);

CREATE INDEX IF NOT EXISTS staging_packet_status_idx
    ON lucidota_go.staging_packet(status, created_at DESC);

-- Seed active terms.
INSERT INTO lucidota_go.term_registry(term, term_number, status)
VALUES
('ENTITY',1,'active'),
('ATTRIBUTE',2,'active'),
('RELATIONSHIP',3,'active'),
('FRICTION',4,'active'),
('LEVERAGE',5,'active'),
('VISIBILITY',6,'active'),
('ACTION',7,'active'),
('EVENT',8,'active'),
('TIME',9,'active'),
('PATTERN',10,'active'),
('HYPOTHESIS',11,'active'),
('CLAIM',12,'active'),
('EVIDENCE',13,'active'),
('ATOMIC_ID',14,'active'),
('SIGNAL',15,'active'),
('GLOW',16,'active'),
('TERM',17,'active'),
('TOOL',18,'active'),
('ALGORITHM',19,'active'),
('NAUGHTY',20,'active'),
('NICE',21,'active'),
('GROUP',22,'active'),
('OPERATOR',23,'active'),
('MODE',24,'active'),
('COMMENT',25,'active'),
('CLUE',26,'active'),
('BOOK',27,'active'),
('HUNCH',28,'active'),
('LEAD',29,'active'),
('SOURCE',30,'active'),
('WITNESS',31,'active'),
('THREAT',32,'active'),
('RUMOUR',33,'active'),
('LOCATION',34,'active'),
('LOSE',35,'active'),
('FIND',36,'active'),
('ANOMALY',37,'active'),
('STORY',38,'active'),
('IDEAL',39,'active'),
('LORE',40,'active'),
('LICENSE',41,'active'),
('REGULATOR',42,'active'),
('GOVERNMENT',43,'active'),
('LAW',44,'active'),
('RULE',45,'active')
ON CONFLICT (term) DO UPDATE SET
    term_number=EXCLUDED.term_number,
    status=EXCLUDED.status,
    updated_at=now();

INSERT INTO lucidota_go.graph_layer(layer, status, description)
VALUES
('digital_world','active','Digital systems, files, accounts, URLs, handles, devices, online traces.'),
('physical_world','active','Physical people, places, objects, addresses, bodies, rooms, buildings.'),
('influence','active','Influence flows and persuasion/control topology.'),
('power','active','Authority, coercive capacity, institutional control, decision rights.'),
('financial','active','Money, assets, liabilities, payments, prices, accounts.'),
('social','active','Social graph, affiliation, community, reputation.'),
('relationship','active','Personal, work, family, lover, enemy, interpersonal ties.'),
('pressure','active','Stress, leverage, coercion, vulnerability, friction.'),
('visibility','active','Public exposure and commonly observable surface.'),
('glow','active','Observer-specific salience / private significance.'),
('map','active','Maps, models, inferred networks, scoped representations.')
ON CONFLICT (layer) DO UPDATE SET
    status=EXCLUDED.status,
    description=EXCLUDED.description,
    updated_at=now();
