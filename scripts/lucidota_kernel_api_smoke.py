#!/usr/bin/env python3
"""CKDOG1 full gRPC API smoke: exercises every current KernelService RPC."""
from __future__ import annotations
import argparse, json, sys, time, socket, tempfile
from pathlib import Path
import grpc

ROOT=Path(__file__).resolve().parents[1]
KERNEL=ROOT/'01_REPOS'/'doggystyle'
sys.path.insert(0, str(KERNEL))
sys.path.insert(0, str(KERNEL/'kernel'/'grpc_gen'))
from kernel.grpc_server import build_server  # noqa:E402
from kernel.grpc_gen import kernel_pb2_grpc  # noqa:E402
import kernel_pb2  # type: ignore # noqa:E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])

def main()->int:
    ap=argparse.ArgumentParser(prog='lucidota-kernel-api-smoke')
    ap.add_argument('--home', type=Path, default=None)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    if args.home is None:
        args.home = Path(tempfile.mkdtemp(prefix='lucidota-kernel-api-smoke.'))
    args.home.mkdir(parents=True, exist_ok=True)
    port=_free_port()
    server=build_server(args.home, f'127.0.0.1:{port}')
    server.start()
    try:
        channel=grpc.insecure_channel(f'127.0.0.1:{port}')
        grpc.channel_ready_future(channel).result(timeout=5)
        stub=kernel_pb2_grpc.KernelServiceStub(channel)
        genesis=stub.GenesisInit(kernel_pb2.GenesisInitRequest(policy_control_mode='SELF_REACTIVE'))
        soul=stub.SoulCreate(kernel_pb2.SoulCreateRequest(soul_uuid=42, term='DIOGENES_TEST'))
        trait=stub.TraitSet(kernel_pb2.TraitSetRequest(soul_uuid=42, trait_index=1, value='INDY_READS'))
        domain=stub.DomainSlotSet(kernel_pb2.DomainSlotSetRequest(
            soul_uuid=42, slot_index=7, target_kind='text', target_text='kernel recall pointer',
            global_symbol='CLAIM_UNVERIFIED', weight_bps=5000, flags=['smoke','recall']))
        shifted=stub.StateShift(kernel_pb2.StateShiftRequest(
            soul_uuid=42, axis=1, intensity_bps=2500, knobs_json=b'{"test":true}'))
        delta=stub.DeltaApply(kernel_pb2.DeltaApplyRequest(
            soul_uuid=42, cycle=int(time.time()), slot_count=4, conversation=b'operator test ingest recall workflow'))
        route=stub.Route(kernel_pb2.RouteRequest(soul_uuid=42, global_symbol='CLAIM_UNVERIFIED', domain_slot=7))
        snap=stub.CreateSnapshot(kernel_pb2.SnapshotRequest(node_ids=['soul:0042']))
        job=stub.SubmitAuthorizedJob(kernel_pb2.AuthorizedJobRequest(
            action='demo_tool_action', endpoint='local://lucidota/demo', payload_json=b'{"ok":true}', idempotency_key='kernel-smoke'))
        route_payload=json.loads(route.route_json.decode('utf-8'))
        delta_payload=json.loads(delta.delta_json.decode('utf-8'))
        report={
            'ok': True,
            'rpcs': ['GenesisInit','SoulCreate','TraitSet','DomainSlotSet','StateShift','DeltaApply','Route','CreateSnapshot','SubmitAuthorizedJob'],
            'ingest_recall': {'delta_hash': delta.delta_hash, 'projection_hash': delta.projection_hash, 'route_trace': route.trace_hash, 'domain_slot': route_payload.get('slot')},
            'graph_state_update': {'trait_version_hash': trait.version_hash, 'domain_version_hash': domain.version_hash, 'shift_version_hash': shifted.version_hash},
            'workflow_tool_action': {'job_id': job.job_id, 'status': job.status},
            'snapshot': {'snapshot_id': snap.snapshot_id, 'root_hash': snap.root_hash},
            'delta_payload_keys': sorted(delta_payload.keys()),
            'genesis_hash': genesis.genesis_hash,
            'soul_uuid': soul.soul_uuid,
        }
        print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))
        return 0
    finally:
        server.stop(grace=0)

if __name__=='__main__':
    raise SystemExit(main())
