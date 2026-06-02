-- 
-- Name: feral_primitive_registry; Type: TABLE; Schema: lucidota_control; Owner: -
--

CREATE TABLE IF NOT EXISTS lucidota_control.feral_primitive_registry (
    primitive_name text PRIMARY KEY,
    category text,
    trigger_pattern jsonb DEFAULT '{}',
    river_feature_key text,
    routing_weight numeric(5,4) DEFAULT 1.0,
    examples text[],
    created_at timestamptz DEFAULT now()
);

INSERT INTO lucidota_control.feral_primitive_registry (primitive_name, category, trigger_pattern, river_feature_key, routing_weight, examples)
VALUES 
('POST_NUT_CLARITY', 'cognitive', '{}', 'focus', 1.0, ARRAY['intense focus', 'sudden clarity']),
('DANGERNOODLE_OUTCAST', 'systemic', '{}', 'threat', 1.0, ARRAY['system-threatening', 'truth-teller']),
('WINDOWS_TAB_SWAMP', 'cognitive', '{}', 'overload', 1.0, ARRAY['cognitive overload', 'information paralysis']),
('ZACK_RAGE', 'emotional', '{}', 'frustration', 1.0, ARRAY['urgent frustration', 'hidden operation']),
('COPYPASTE_PROGRAMMER', 'behavioral', '{}', 'template', 1.0, ARRAY['template without understanding', 'mindless repetition']),
('QUIET_HINGE', 'procedural', '{}', 'procedural', 1.0, ARRAY['boring procedural move', 'high cascade potential'])
ON CONFLICT (primitive_name) DO NOTHING;

-- 
-- Name: response_phenotype_registry; Type: TABLE; Schema: lucidota_control; Owner: -
--

CREATE TABLE IF NOT EXISTS lucidota_control.response_phenotype_registry (
    phenotype_name text PRIMARY KEY,
    description text,
    trigger_conditions jsonb DEFAULT '{}',
    output_structure jsonb DEFAULT '{}',
    river_training_hook text,
    created_at timestamptz DEFAULT now()
);

INSERT INTO lucidota_control.response_phenotype_registry (phenotype_name, description, trigger_conditions, output_structure, river_training_hook)
VALUES 
('ZACK_LEARN_MODE', 'learning mode', '{}', '{}', 'learn'),
('ABDUCTIVE_DECODE', 'abductive decoding', '{}', '{}', 'decode'),
('QUIET_HINGE_MOVE', 'quiet hinge move', '{}', '{}', 'move'),
('FAIRNESS_TRIGGER', 'fairness trigger', '{}', '{}', 'trigger'),
('CASCADE_MAP', 'cascade map', '{}', '{}', 'map')
ON CONFLICT (phenotype_name) DO NOTHING;
