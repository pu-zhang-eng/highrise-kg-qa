"""Load data/neo4j_data/knowledge_graph.csv into Neo4j.

Every entity becomes an (:Entity {name}) node and every relation value becomes its
own relationship type, e.g.
    (:Entity {name: '承载能力'})-[:满足]->(:Entity {name: '结构变形要求'})
which is the shape qa_app.py queries.

Connection settings come from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD (see README).
Safe to re-run: MERGE does not create duplicates.
"""
import csv
import os
import sys
from collections import defaultdict

from py2neo import Graph

CSV_PATH = os.path.join("data", "neo4j_data", "knowledge_graph.csv")


def load_triples(path):
    """Group (subject, object) pairs by relation, since each relation is its own type."""
    groups = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            s, r, o = row["Subject"].strip(), row["Relation"].strip(), row["Object"].strip()
            if s and r and o:
                groups[r].append({"s": s, "o": o})
    return groups


def main():
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        sys.exit("Set NEO4J_PASSWORD first (see README).")
    graph = Graph(os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                  auth=(os.getenv("NEO4J_USER", "neo4j"), password))

    try:
        graph.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
    except Exception as e:  # older Neo4j versions use other syntax; the index only speeds up MERGE
        print(f"Index not created ({e}); continuing without it.")

    groups = load_triples(CSV_PATH)
    total = 0
    for rel_type, rows in groups.items():
        # A relationship type cannot be a query parameter, so it is quoted with backticks.
        safe_type = rel_type.replace("`", "``")
        graph.run(
            "UNWIND $rows AS row "
            "MERGE (s:Entity {name: row.s}) "
            "MERGE (o:Entity {name: row.o}) "
            f"MERGE (s)-[:`{safe_type}`]->(o)",
            rows=rows,
        )
        total += len(rows)
    print(f"Imported {total} triples across {len(groups)} relation types.")


if __name__ == "__main__":
    main()
