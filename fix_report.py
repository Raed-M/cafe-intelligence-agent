import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.reporting.report_generator import generate_report

conn = sqlite3.connect('db/checkpoints.sqlite')
checkpointer = SqliteSaver(conn)
state = checkpointer.get_tuple({'configurable': {'thread_id': 'run_4ba8b7f730d6'}})

if state:
    channel_values = state.checkpoint['channel_values']
    print("Generating report...")
    result = generate_report(channel_values)
    print("Report generated:", result)
else:
    print("Checkpoint not found!")

