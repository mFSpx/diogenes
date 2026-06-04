CREATE OR REPLACE FUNCTION lucidota_control.lucidota_wake_workers() 
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('lucidota_queue_wakeup', TG_TABLE_NAME);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_wake_absurd_queue_job ON lucidota_control.absurd_queue_job;
CREATE TRIGGER trg_wake_absurd_queue_job
AFTER INSERT ON lucidota_control.absurd_queue_job 
FOR EACH STATEMENT 
EXECUTE FUNCTION lucidota_control.lucidota_wake_workers();
