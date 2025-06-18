import sys
from pathlib import Path

# Add project root to sys.path to allow a_s_b_absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

"""
Step 2: Generate Demo Traces
Purpose: Generate 10 high-quality demo traces covering different dietary scenarios
"""

import pandas as pd
import warnings
from pathlib import Path
from rich.console import Console
from rich.progress import track
from backend.utils import get_agent_response

# Suppress pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

console = Console()

def load_dietary_queries():
    """Load dietary queries from the HW3 dataset."""
    queries_path = Path("homeworks/hw3/data/dietary_queries.csv")
    
    if not queries_path.exists():
        console.print(f"[red]Error: {queries_path} not found![/red]")
        return None
    
    df = pd.read_csv(queries_path)
    return df

def select_demo_queries(df):
    """Select 4 demo queries that have previously produced FAIL examples."""
    
    # Select queries that have produced FAIL examples in labeled_traces.csv
    # This increases likelihood of getting interesting PASS/FAIL variety for demo
    selected_ids = [26, 43, 46, 48]  # sugar-free, vegetarian, paleo, gluten-free
    
    selected_queries = []
    for query_id in selected_ids:
        query_row = df[df['id'] == query_id]
        if not query_row.empty:
            selected_queries.append({
                'id': query_row.iloc[0]['id'],
                'query': query_row.iloc[0]['query'],
                'dietary_restriction': query_row.iloc[0]['dietary_restriction']
            })
    
    return selected_queries

def generate_trace(query_data):
    """Generate a single trace by sending query through the recipe bot."""
    
    query = query_data['query']
    dietary_restriction = query_data['dietary_restriction']
    
    try:
        # Send query through recipe bot with dietary restriction metadata
        messages = [{"role": "user", "content": query}]
        metadata = {
            "dietary_restriction": dietary_restriction,
            "response_number": query_data.get('response_number', 1)
        }
        response_messages = get_agent_response(messages, metadata)
        
        # Extract the assistant's response
        assistant_response = ""
        for msg in response_messages:
            if msg["role"] == "assistant":
                assistant_response = msg["content"]
                break
        
        return {
            'query_id': query_data['id'],
            'query': query,
            'dietary_restriction': dietary_restriction,
            'response': assistant_response,
            'success': True
        }
        
    except Exception as e:
        return {
            'query_id': query_data['id'],
            'query': query,
            'dietary_restriction': dietary_restriction,
            'response': "",
            'success': False,
            'error': str(e)
        }

def main():
    """Main function to generate demo traces."""
    console.print("[bold blue]Step 2: Generate Demo Traces[/bold blue]")
    console.print("=" * 50)
    
    # Load queries
    df = load_dietary_queries()
    if df is None:
        return False
    
    console.print(f"[green]Loaded {len(df)} dietary queries[/green]")
    
    # Select demo queries
    demo_queries = select_demo_queries(df)
    console.print(f"[green]Selected {len(demo_queries)} queries for demo[/green]")
    
    # Show selected queries
    console.print("\n[bold]Selected Demo Queries:[/bold]")
    for i, query in enumerate(demo_queries, 1):
        console.print(f"{i}. [{query['dietary_restriction']}] {query['query'][:60]}...")
    
    # Generate traces (3 responses per query)
    total_traces = len(demo_queries) * 3
    console.print(f"\n[yellow]Generating {total_traces} traces ({len(demo_queries)} queries × 3 responses each)...[/yellow]")
    console.print("[blue]Each trace will be sent to Langfuse automatically[/blue]")
    
    traces = []
    all_generations = []
    
    # Create list of all generations to process
    for query_data in demo_queries:
        for i in range(3):
            generation_data = query_data.copy()
            generation_data['response_number'] = i + 1
            all_generations.append(generation_data)
    
    for generation_data in track(all_generations, description="Generating traces"):
        trace = generate_trace(generation_data)
        traces.append(trace)
    
    # Summary
    successful_traces = [t for t in traces if t['success']]
    failed_traces = [t for t in traces if not t['success']]
    
    console.print("\n[bold]Generation Summary:[/bold]")
    console.print(f"[green]✅ Successful traces: {len(successful_traces)}[/green]")
    console.print(f"[red]❌ Failed traces: {len(failed_traces)}[/red]")
    
    if failed_traces:
        console.print("\n[red]Failed traces:[/red]")
        for trace in failed_traces:
            console.print(f"- {trace['dietary_restriction']}: {trace.get('error', 'Unknown error')}")
    
    # Show dietary restrictions covered
    restrictions_covered = set(t['dietary_restriction'] for t in successful_traces)
    console.print(f"\n[blue]Dietary restrictions covered: {', '.join(sorted(restrictions_covered))}[/blue]")
    
    console.print("\n[bold green]Step 2 Complete! ✅[/bold green]")
    console.print(f"[blue]Check your Langfuse dashboard to see the {len(successful_traces)} traces[/blue]")
    console.print("[yellow]Each trace should have a descriptive name like 'Query: I'm vegan but...'[/yellow]")
    
    return len(successful_traces) > 0

if __name__ == "__main__":
    success = main()
    if not success:
        console.print("[red]Demo trace generation failed![/red]")
        exit(1)
    else:
        console.print("[green]Demo traces generated successfully![/green]")
