# app.py — Interfaccia Streamlit completa per Visual SQL Builder

import streamlit as st
import json
import tempfile
import html
import csv
import io
from collections import defaultdict
from pyvis.network import Network

from core import update_output_columns, generate_sql_from_graph, infer_custom_sql_columns_with_llm


def extract_columns_from_csv(uploaded_file):
    """Legge la prima riga del CSV e restituisce i nomi delle colonne."""
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        reader = csv.reader(io.StringIO(content))
        return [c.strip() for c in next(reader, []) if c.strip()]
    except Exception:
        return []

# =========================
# Config iniziale
# =========================
st.set_page_config(layout="wide")
st.title("Visual SQL Builder")

if 'data' not in st.session_state:
    st.session_state['data'] = {"nodes": [], "edges": []}
    st.session_state['node_counter'] = 1


# =========================
# Sidebar: Add Node
# =========================
st.sidebar.header("Add Node")
node_type = st.sidebar.selectbox("Node type", [
    "input_table", "filter", "select", "join", "group_by", "sort",
    "limit", "arithmetic", "case", "window", "custom_sql"
])

with st.sidebar.form("add_node_form"):
    alias = st.text_input("Alias")
    note = st.text_area("Note", height=70)
    extra = {}

    if node_type == "input_table":
        extra["table"] = st.text_input("Table name")
        uploaded = st.file_uploader("CSV sample (header only)", type="csv", key="add_csv")
        manual_cols = st.text_input("Columns (comma-separated)", disabled=uploaded is not None)
        if uploaded is not None:
            extra["columns"] = extract_columns_from_csv(uploaded)
        else:
            extra["columns"] = manual_cols
    elif node_type == "filter":
        extra["condition"] = st.text_input("WHERE condition")
    elif node_type == "select":
        extra["columns"] = st.text_input("Columns (comma-separated)")
    elif node_type == "join":
        extra["on"] = st.text_input("Join key")
        extra["how"] = st.selectbox("Join type", ["inner", "left", "right", "full"])
        extra["using"] = st.checkbox("Use USING")
    elif node_type == "group_by":
        extra["group_columns"] = st.text_input("Group by columns")
        extra["agg_columns"] = st.text_area("Aggregates (one per line)").splitlines()
    elif node_type in ["window", "case"]:
        extra["expression"] = st.text_area("Expression", height=100)
    elif node_type == "custom_sql":
        extra["sql"] = st.text_area("SQL code", height=150)

    declared = st.text_input("Override output columns (comma-separated) [optional]")
    lock_override = st.checkbox("Lock output override", value=False)

    submit_node = st.form_submit_button("Add Node")
    if submit_node:
        nid = str(st.session_state['node_counter'])
        node = {"id": nid, "type": node_type, "alias": alias, "note": note}
        node.update(extra)

        if "columns" in node and isinstance(node["columns"], str):
            node["columns"] = [x.strip() for x in node["columns"].split(",") if x.strip()]
        if "group_columns" in node and isinstance(node["group_columns"], str):
            node["group_columns"] = [x.strip() for x in node["group_columns"].split(",") if x.strip()]

        node["declared_output_columns"] = [x.strip() for x in declared.split(",") if x.strip()]
        node["lock_output_columns"] = lock_override

        st.session_state['data']['nodes'].append(node)
        st.session_state['node_counter'] += 1
        update_output_columns(st.session_state['data'])
        st.rerun()


# =========================
# Sidebar: Add Edge
# =========================
if len(st.session_state['data']['nodes']) >= 2:
    st.sidebar.header("Add Edge")
    node_ids = [n['id'] for n in st.session_state['data']['nodes']]
    with st.sidebar.form("add_edge_form"):
        from_id = st.selectbox("From", node_ids, key="from")
        to_id = st.selectbox("To", node_ids, key="to")
        submit_edge = st.form_submit_button("Add Edge")
        if submit_edge:
            st.session_state['data']['edges'].append({"from": from_id, "to": to_id})
            update_output_columns(st.session_state['data'])
            st.rerun()


# =========================
# JSON Editor
# =========================
st.subheader("1. JSON Graph Structure")
json_str = st.text_area("Edit Graph JSON", json.dumps(st.session_state['data'], indent=2), height=400)
try:
    parsed = json.loads(json_str)
    st.session_state['data'] = parsed
    update_output_columns(st.session_state['data'])
    st.success("Valid JSON")
except json.JSONDecodeError:
    st.error("Invalid JSON")


# =========================
# Graph Visualization
# =========================
def render_graph(data):
    g = Network(height='500px', width='100%', directed=True)
    g.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=250)

    color_map = {
        "input_table": "#ADD8E6", "filter": "#90EE90", "select": "#FFD700",
        "join": "#FFB6C1", "group_by": "#D8BFD8", "sort": "#FFA07A",
        "limit": "#B0C4DE", "arithmetic": "#F4A460", "case": "#87CEFA",
        "window": "#9370DB", "custom_sql": "#C0C0C0"
    }

    depth = defaultdict(int)
    for edge in data['edges']:
        depth[edge['to']] = max(depth[edge['to']], depth[edge['from']] + 1)

    layer_nodes = defaultdict(list)
    for node in data['nodes']:
        layer_nodes[depth[node['id']]].append(node)

    for d, nodes in layer_nodes.items():
        for i, node in enumerate(nodes):
            label = f"{node['type']}\nID: {node['id']}"
            if node.get("alias"): label += f"\nAs: {node['alias']}"
            if node.get("note"): label += f"\nNote: {node['note']}"
            preview = node.get("output_columns", [])
            if preview: label += f"\nCols: {', '.join(preview[:5])}"

            g.add_node(
                node['id'],
                label=label,
                title=html.escape(node.get("note", "")),
                color=color_map.get(node['type'], "#D3D3D3"),
                x=d*250,
                y=i*150,
                physics=False
            )

    for edge in data['edges']:
        g.add_edge(edge['from'], edge['to'])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    g.write_html(tmp.name)
    return tmp.name


st.subheader("2. Graph View")
html_file = render_graph(st.session_state['data'])
st.components.v1.html(open(html_file).read(), height=550, scrolling=True)


# =========================
# Node Editor + LLM Inferenza
# =========================
st.subheader("3. Inspect/Edit Node")
node_ids = [n['id'] for n in st.session_state['data']['nodes']]
if node_ids:
    selected = st.selectbox("Select node ID", node_ids)
    selected_node = next((n for n in st.session_state['data']['nodes'] if n['id'] == selected), {})

    with st.form("edit_node"):
        st.markdown(f"### Node {selected_node['id']} ({selected_node['type']})")

        selected_node['alias'] = st.text_input("Alias", selected_node.get("alias", ""))
        selected_node['note'] = st.text_area("Note", selected_node.get("note", ""))

        if selected_node['type'] == "input_table":
            selected_node['table'] = st.text_input("Table name", selected_node.get("table", ""))
            uploaded_edit = st.file_uploader(
                "CSV sample (header only)", type="csv", key=f"edit_csv_{selected}"
            )
            cols = ", ".join(selected_node.get("columns", []))
            manual_edit = st.text_input(
                "Columns", cols, disabled=uploaded_edit is not None
            )
            if uploaded_edit is not None:
                selected_node['columns'] = extract_columns_from_csv(uploaded_edit)
            else:
                selected_node['columns'] = [c.strip() for c in manual_edit.split(",") if c.strip()]

        if selected_node['type'] == "custom_sql":
            selected_node['sql'] = st.text_area("SQL code", selected_node.get("sql", ""), height=150)

        override_now = ", ".join(selected_node.get("declared_output_columns", []))
        selected_node["declared_output_columns"] = [x.strip() for x in st.text_input("Override columns", override_now).split(",") if x.strip()]
        selected_node["lock_output_columns"] = st.checkbox("Lock override", selected_node.get("lock_output_columns", False))

        if selected_node['type'] == "custom_sql":
            if st.form_submit_button("Infer columns with LLM"):
                if selected_node["lock_output_columns"]:
                    st.warning("Unlock override to infer columns.")
                elif not selected_node.get("sql"):
                    st.warning("No SQL to analyze.")
                else:
                    parents = [e['from'] for e in st.session_state['data']['edges'] if e['to'] == selected_node['id']]
                    parent_sources = []
                    for pid in parents:
                        pn = next((n for n in st.session_state['data']['nodes'] if n['id'] == pid), None)
                        if pn:
                            parent_sources.append({
                                "node_id": pid,
                                "alias": pn.get("alias"),
                                "table": pn.get("table") if pn.get("type") == "input_table" else None,
                                "columns": pn.get("output_columns", [])
                            })
                    try:
                        with st.spinner("Inferring..."):
                            inferred = infer_custom_sql_columns_with_llm(
                                sql_text=selected_node["sql"],
                                parent_sources=parent_sources
                            )
                        if inferred:
                            selected_node["declared_output_columns"] = inferred
                            selected_node["lock_output_columns"] = True
                            st.success("Columns inferred.")
                            update_output_columns(st.session_state['data'])
                            st.rerun()
                        else:
                            st.warning("No columns returned.")
                    except Exception as e:
                        st.error(f"LLM error: {e}")

        save_btn = st.form_submit_button("Save")
        del_btn = st.form_submit_button("Delete")
        if save_btn:
            update_output_columns(st.session_state['data'])
            st.rerun()
        if del_btn:
            st.session_state['data']['nodes'] = [n for n in st.session_state['data']['nodes'] if n['id'] != selected]
            st.session_state['data']['edges'] = [e for e in st.session_state['data']['edges'] if e['from'] != selected and e['to'] != selected]
            update_output_columns(st.session_state['data'])
            st.rerun()
else:
    st.info("No nodes to edit.")


# =========================
# SQL Generation
# =========================
st.subheader("4. Generated SQL")
if st.button("Generate SQL"):
    sql = generate_sql_from_graph(st.session_state['data'])
    st.code(sql, language="sql")

st.download_button("Download JSON", json.dumps(st.session_state['data'], indent=2), file_name="flow.json")
