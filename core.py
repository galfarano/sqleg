# core.py — Logica e utilità per Visual SQL Builder

import re
import json
from collections import defaultdict, deque
import os
from dotenv import load_dotenv

load_dotenv()
my_key = os.getenv("OPENAI_API_KEY")
if not my_key:
    raise ValueError("Chiave API mancante. Sei sicuro di avere un file .env con OPENAI_API_KEY?")


# === Utils ===

def parse_agg_name(expr):
    match = re.search(r'\bas\s+(\w+)', expr, re.IGNORECASE)
    if match:
        return match.group(1)
    return expr.split('(')[0].strip() + "_val"

def _parse_llm_columns(text: str):
    def _from_obj(obj):
        if isinstance(obj, dict):
            cols = obj.get("columns", obj.get("output_columns", obj.get("fields")))
            if isinstance(cols, list):
                return [c["name"] if isinstance(c, dict) and "name" in c else str(c) for c in cols]
        if isinstance(obj, list):
            return [c["name"] if isinstance(c, dict) and "name" in c else str(c) for c in obj]
        return []

    try:
        obj = json.loads(text)
        cols = _from_obj(obj)
        if cols:
            return [str(c).strip() for c in cols if str(c).strip()]
    except Exception:
        pass

    import re as _re
    m = _re.search(r'(\{.*\}|\[.*\])', text, _re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            cols = _from_obj(obj)
            if cols:
                return [str(c).strip() for c in cols if str(c).strip()]
        except Exception:
            pass

    m2 = re.search(r'\[([^\]]+)\]', text)
    if m2:
        raw = m2.group(1)
        items = [x.strip().strip('",\'') for x in raw.split(",")]
        return [x for x in items if x]
    return []

# === Inferenza via OpenAI ===

def get_openai_client():
    try:
        from openai import OpenAI
        import os
        key = ""
        if not key:
            raise RuntimeError("OPENAI_API_KEY non configurata.")
        return OpenAI(api_key=key)
    except Exception as e:
        raise RuntimeError(f"Errore client OpenAI: {e}")

def infer_custom_sql_columns_with_llm(sql_text: str, parent_sources: list, dialect: str = "generic"):
    client = get_openai_client()

    payload = {
        "task": "infer_output_columns",
        "dialect": dialect,
        "parents": parent_sources,
        "sql": sql_text
    }

    messages = [
        {"role": "system", "content": (
            "Sei un assistente che deduce lo schema di output di una query SQL.\n"
            "Rispondi SOLO con JSON valido nel formato {\"columns\": [\"col1\", ...]}"
        )},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=messages,
        response_format={"type": "json_object"}
    )
    response_text = completion.choices[0].message.content
    cols = _parse_llm_columns(response_text)
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]

# === Colonna Propagation ===

def infer_output_columns(node, parent_inputs: dict):
    declared = [c.strip() for c in node.get("declared_output_columns", []) if str(c).strip()]
    if declared:
        return declared

    ntype = node["type"]

    if ntype == "input_table":
        return node.get("columns", [])

    elif ntype in ["filter", "limit", "sort", "window", "case", "arithmetic"]:
        return next(iter(parent_inputs.values()), [])

    elif ntype == "select":
        return node.get("columns", [])

    elif ntype == "group_by":
        return node.get("group_columns", []) + [parse_agg_name(a) for a in node.get("agg_columns", [])]

    elif ntype == "join":
        combined = []
        for cols in parent_inputs.values():
            combined.extend(cols)
        seen = set(); return [c for c in combined if not (c in seen or seen.add(c))]

    elif ntype == "custom_sql":
        return declared

    return declared

def update_output_columns(graph_data):
    nodes = {n["id"]: n for n in graph_data["nodes"]}
    edges = graph_data["edges"]

    children = defaultdict(list)
    in_degrees = defaultdict(int)

    for edge in edges:
        src, tgt = edge["from"], edge["to"]
        children[src].append(tgt)
        in_degrees[tgt] += 1

    queue = deque([nid for nid in nodes if in_degrees[nid] == 0])
    sorted_ids = []

    while queue:
        current = queue.popleft()
        sorted_ids.append(current)
        for child in children[current]:
            in_degrees[child] -= 1
            if in_degrees[child] == 0:
                queue.append(child)

    input_columns_by_id = {}
    for nid in sorted_ids:
        node = nodes[nid]
        parent_ids = [e["from"] for e in edges if e["to"] == nid]
        parent_inputs = {pid: input_columns_by_id.get(pid, []) for pid in parent_ids}
        node["output_columns"] = infer_output_columns(node, parent_inputs)
        input_columns_by_id[nid] = node["output_columns"]

# === SQL Generator ===

def generate_sql_from_graph(graph_json):
    nodes = {node['id']: node for node in graph_json['nodes']}
    edges = graph_json['edges']
    children = defaultdict(list)
    in_degrees = defaultdict(int)

    for edge in edges:
        children[edge['from']].append(edge['to'])
        in_degrees[edge['to']] += 1

    queue = deque([nid for nid in nodes if in_degrees[nid] == 0])
    sorted_ids = []

    while queue:
        nid = queue.popleft()
        sorted_ids.append(nid)
        for child in children[nid]:
            in_degrees[child] -= 1
            if in_degrees[child] == 0:
                queue.append(child)

    alias_map = {}
    sql_parts = []

    for nid in sorted_ids:
        node = nodes[nid]
        alias = node.get("alias")
        parents = [e['from'] for e in edges if e['to'] == nid]
        inputs = [alias_map.get(pid, "UNKNOWN_INPUT") for pid in parents]

        note = f"-- {node['note']}" if node.get("note") else ""
        ntype = node["type"]
        body = "-- unknown"

        if ntype == "input_table":
            body = f"SELECT * FROM {node.get('table', 'table_name')}"

        elif ntype == "select":
            cols = ", ".join(node.get("columns", ["*"]))
            body = f"SELECT {cols} FROM {inputs[0]}"

        elif ntype == "filter":
            cond = node.get("condition", "1=1")
            body = f"SELECT * FROM {inputs[0]} WHERE {cond}"

        elif ntype == "group_by":
            gc = ", ".join(node.get("group_columns", []))
            aggs = ", ".join(node.get("agg_columns", []))
            sep = ", " if gc and aggs else ""
            body = f"SELECT {gc}{sep}{aggs} FROM {inputs[0]} GROUP BY {gc}" if gc else f"SELECT {aggs} FROM {inputs[0]}"

        elif ntype == "join":
            how = node.get("how", "inner").upper()
            on = node.get("on", "1=1")
            if node.get("using"):
                body = f"SELECT * FROM {inputs[0]} {how} JOIN {inputs[1]} USING ({on})"
            else:
                body = f"SELECT * FROM {inputs[0]} {how} JOIN {inputs[1]} ON {on}"

        elif ntype == "custom_sql":
            body = node.get("sql", "-- empty SQL")

        elif ntype == "arithmetic":
            expr = node.get("expression", "1+1")
            body = f"SELECT *, {expr} AS computed FROM {inputs[0]}"

        elif ntype == "case":
            expr = node.get("expression", "CASE WHEN TRUE THEN 1 END")
            body = f"SELECT *, {expr} AS case_result FROM {inputs[0]}"

        elif ntype == "limit":
            body = f"SELECT * FROM {inputs[0]} LIMIT {node.get('limit', 10)}"

        elif ntype == "sort":
            body = f"SELECT * FROM {inputs[0]} ORDER BY {node.get('by', '1')}"

        elif ntype == "window":
            expr = node.get("expression", "ROW_NUMBER() OVER()")
            body = f"SELECT *, {expr} AS win_result FROM {inputs[0]}"

        if alias:
            sql_parts.append(f"{note}\n{alias} AS (\n  {body}\n)")
            alias_map[nid] = alias
        else:
            alias_map[nid] = inputs[-1] if inputs else "UNKNOWN"

    final = alias_map.get(sorted_ids[-1], "UNKNOWN")
    return "WITH\n" + ",\n".join(sql_parts) + f"\nSELECT * FROM {final};"
