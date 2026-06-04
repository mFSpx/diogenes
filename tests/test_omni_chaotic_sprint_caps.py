from ALGOS.omni_chaotic_sprint import safe_fetchall


class CursorNoFetchAll:
    def __init__(self):
        self.calls = 0
        self.batches = [[{"i": 1}], [{"i": 2}], []]

    def fetchmany(self, n):
        self.calls += 1
        return self.batches.pop(0)

    def fetchall(self):  # pragma: no cover - must not be called
        raise AssertionError("fetchall materializes unbounded result sets")


class Conn:
    def __init__(self):
        self.cursor = CursorNoFetchAll()

    def execute(self, sql, params):
        return self.cursor


def test_safe_fetchall_uses_fetchmany_batches_not_fetchall():
    conn = Conn()
    assert safe_fetchall(conn, "select 1", batch_size=1, max_rows=10) == [{"i": 1}, {"i": 2}]
    assert conn.cursor.calls == 3
