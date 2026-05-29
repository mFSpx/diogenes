from pathlib import Path
import py_compile


def test_graph_edge_materializer_records_helper_receipt():
    text = Path('scripts/graph_edge_materialize.py').read_text(encoding='utf-8')
    assert '065_graph_materialization_helper_v2.sql' in text
    assert 'graph_materialization_helper_receipt' in text
    assert 'helper_receipt_uuid' in text
    assert "result.update" in text and "helper_receipt_uuid" in text


def test_graph_edge_materializer_is_importable_python():
    py_compile.compile('scripts/graph_edge_materialize.py', doraise=True)
