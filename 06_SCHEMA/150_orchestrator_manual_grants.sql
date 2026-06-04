-- PostgREST read grants for manual/orchestrator truth surfaces.

GRANT SELECT ON
    lucidota_canon.manual_current,
    lucidota_canon.canon_current,
    lucidota_canon.active_goal,
    lucidota_canon.api_workflow_registry,
    lucidota_canon.flow_specs,
    lucidota_canon.flow_receipts,
    lucidota_canon.capability_registry,
    lucidota_canon.model_registry,
    lucidota_canon.provider_registry,
    lucidota_canon.workflow_registry,
    lucidota_canon.daemon_status
TO lucidota_postgrest_anon, mfspx;
