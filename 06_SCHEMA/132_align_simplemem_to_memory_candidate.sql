-- 
-- Name: 132_align_simplemem_to_memory_candidate; Type: MIGRATION; Schema: lucidota_control; Owner: -
--

ALTER TABLE lucidota_control.simplemem_candidate
ADD COLUMN IF NOT EXISTS claim_id UUID;

ALTER TABLE lucidota_control.simplemem_candidate
DROP CONSTRAINT IF EXISTS chk_simplemem_candidate_status;

ALTER TABLE lucidota_control.simplemem_candidate
ADD CONSTRAINT chk_simplemem_candidate_status
CHECK (status IN ('CANDIDATE','ACCEPTED','REJECTED'));
