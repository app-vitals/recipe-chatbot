#!/usr/bin/env python3
"""
Step 4: Create Dataset for Experiments
Purpose: Extract demo traces and create a Langfuse dataset for judge prompt experiments
"""

from dotenv import load_dotenv
from langfuse import Langfuse
from rich.console import Console
from rich.progress import track

load_dotenv(override=False)

console = Console()

def get_demo_traces():
    """Get the demo traces we generated."""
    try:
        langfuse = Langfuse()
        
        console.print("[yellow]Fetching demo traces...[/yellow]")
        
        # Get recent traces with our naming pattern
        traces = langfuse.fetch_traces(
            limit=20,
            order_by="timestamp.desc", 
        )
        
        # Filter for our demo traces
        demo_traces = []
        for trace in traces.data:
            if (trace.name and trace.name.startswith("Query:")):
                demo_traces.append(trace)
                if len(demo_traces) >= 12:  # Our 4 queries × 3 responses
                    break
        
        console.print(f"[green]Found {len(demo_traces)} demo traces[/green]")
        return demo_traces
        
    except Exception as e:
        console.print(f"[red]Error fetching traces: {str(e)}[/red]")
        return []

def extract_trace_data(trace):
    """Extract query, dietary_restriction, and response from a trace."""
    try:
        # Extract query from user message 
        query = ""
        if hasattr(trace, 'input') and trace.input:
            if isinstance(trace.input, dict) and 'messages' in trace.input:
                messages = trace.input['messages']
                # Find the user message (query)
                for msg in messages:
                    if msg.get('role') == 'user':
                        query = msg.get('content', '')
                        break
        
        # Get response from trace output
        response = ""
        if hasattr(trace, 'output') and trace.output and isinstance(trace.output, dict):
            response = trace.output.get('content', '')
        
        # Get dietary restriction from observations metadata (since that's where it's stored)
        dietary_restriction = ""
        if hasattr(trace, 'observations') and trace.observations:
            # Get the first observation to check for metadata
            obs_id = trace.observations[0] if trace.observations else None
            if obs_id:
                try:
                    langfuse = Langfuse()
                    observation = langfuse.get_observation(obs_id)
                    if hasattr(observation, 'metadata') and observation.metadata:
                        dietary_restriction = observation.metadata.get('dietary_restriction', '')
                except:
                    # If we can't fetch the observation, try to extract from trace name or other sources
                    pass
        
        # Fallback: try to infer dietary restriction from query content
        if not dietary_restriction and query:
            query_lower = query.lower()
            if 'gluten' in query_lower:
                dietary_restriction = 'gluten-free'
            elif 'sugar-free' in query_lower:
                dietary_restriction = 'sugar-free'
            elif 'vegetarian' in query_lower:
                dietary_restriction = 'vegetarian'
            elif 'paleo' in query_lower:
                dietary_restriction = 'paleo'
        
        return {
            'query': query,
            'dietary_restriction': dietary_restriction,
            'response': response,
            'trace_id': trace.id
        }
        
    except Exception as e:
        console.print(f"[yellow]Error extracting data from trace {trace.id}: {str(e)}[/yellow]")
        return None

def create_langfuse_dataset(dataset_items):
    """Create a Langfuse dataset from extracted trace data."""
    try:
        langfuse = Langfuse()
        
        console.print("[yellow]Creating Langfuse dataset...[/yellow]")
        
        # Create dataset
        dataset = langfuse.create_dataset(
            name="Dietary Adherence Demo",
            description="Demo traces for testing LLM-as-a-Judge prompts on dietary adherence evaluation"
        )
        
        console.print(f"[green]Created dataset: {dataset.name}[/green]")
        
        # Add items to dataset
        for item in track(dataset_items, description="Adding items to dataset"):
            if item and item['response']:  # Only add items with valid responses
                langfuse.create_dataset_item(
                    dataset_name=dataset.name,
                    input={
                        "query": item['query'],
                        "dietary_restriction": item['dietary_restriction'],
                        "response": item['response']
                    },
                    expected_output=None,  # No expected output for prompt comparison
                    metadata={
                        "source_trace_id": item['trace_id'],
                        "dietary_restriction": item['dietary_restriction']
                    }
                )
        
        console.print(f"[green]✅ Added {len([i for i in dataset_items if i and i['response']])} items to dataset[/green]")
        console.print(f"[blue]Dataset ID: {dataset.id}[/blue]")
        
        return dataset
        
    except Exception as e:
        console.print(f"[red]Error creating dataset: {str(e)}[/red]")
        return None

def show_dataset_preview(dataset_items):
    """Show a preview of the dataset items."""
    console.print("\n[bold]Dataset Preview:[/bold]")
    console.print("=" * 80)
    
    valid_items = [item for item in dataset_items if item and item['response']]
    
    for i, item in enumerate(valid_items[:3], 1):  # Show first 3 items
        console.print(f"\n[cyan]Item {i}:[/cyan]")
        console.print(f"[blue]Query:[/blue] {item['query'][:60]}...")
        console.print(f"[blue]Dietary Restriction:[/blue] {item['dietary_restriction']}")
        console.print(f"[blue]Response:[/blue] {item['response'][:100]}...")
        console.print(f"[blue]Trace ID:[/blue] {item['trace_id']}")
    
    if len(valid_items) > 3:
        console.print(f"\n[yellow]... and {len(valid_items) - 3} more items[/yellow]")

def main():
    """Main function to create dataset from demo traces."""
    console.print("[bold blue]Step 4: Create Dataset for Experiments[/bold blue]")
    console.print("=" * 50)
    
    # Get demo traces
    traces = get_demo_traces()
    if not traces:
        console.print("[red]No demo traces found. Please run Step 2 first.[/red]")
        return False
    
    # Extract data from traces
    console.print("[yellow]Extracting data from traces...[/yellow]")
    dataset_items = []
    
    for trace in traces:
        item = extract_trace_data(trace)
        dataset_items.append(item)
    
    # Filter out invalid items
    valid_items = [item for item in dataset_items if item and item['response']]
    console.print(f"[green]Extracted {len(valid_items)} valid dataset items[/green]")
    
    if len(valid_items) == 0:
        console.print("[red]No valid items found. Check trace structure.[/red]")
        return False
    
    # Show preview
    show_dataset_preview(dataset_items)
    
    # Create Langfuse dataset
    dataset = create_langfuse_dataset(valid_items)
    if not dataset:
        return False
    
    console.print("\n[bold green]Step 4 Complete! ✅[/bold green]")
    console.print("[blue]Dataset created successfully in Langfuse[/blue]")
    console.print("[yellow]You can now create experiments using this dataset[/yellow]")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        console.print("[red]Dataset creation failed![/red]")
        exit(1)
    else:
        console.print("[green]Dataset created successfully![/green]")
