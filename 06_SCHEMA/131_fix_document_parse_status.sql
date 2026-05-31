-- Fix document parse status check constraint
ALTER TABLE lucidota_korpus.document_parse_run DROP CONSTRAINT IF EXISTS chk_document_parse_run_status;
ALTER TABLE lucidota_korpus.document_parse_run ADD CONSTRAINT chk_document_parse_run_status CHECK (status IN ('PARSED','OCR_REQUIRED','OCR_BLOCKED','UNSUPPORTED','FAILED'));
