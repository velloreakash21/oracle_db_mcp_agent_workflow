"""
Code Assistant - Main Entry Point
An AI-powered assistant for finding documentation and code examples.
"""
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.telemetry.tracing import init_telemetry
from src.agents.orchestrator import ask_assistant


def main():
    """Main function - interactive code assistant."""

    # Initialize telemetry
    init_telemetry()

    print("="*60)
    print("  CODE ASSISTANT")
    print("  Your AI-powered coding helper")
    print("="*60)
    print("\nI can help you with:")
    print("  - Finding documentation and explanations")
    print("  - Code examples from our database")
    print("  - Best practices and tutorials")
    print("\nType 'quit' or 'exit' to stop.")
    print("-"*60)

    while True:
        try:
            query = input("\nYour question: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! Happy coding!")
                break

            print("\nSearching...")
            response = ask_assistant(query)
            print("\n" + "-"*60)
            print(response)
            print("-"*60)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


def single_query(query: str):
    """Run a single query (for scripting/testing)."""
    init_telemetry()
    response = ask_assistant(query)
    print(response)
    return response


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run single query from command line
        query = " ".join(sys.argv[1:])
        single_query(query)
    else:
        # Interactive mode
        main()
