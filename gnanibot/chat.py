"""Terminal UI for the Gnani.ai bot. All logic lives in rag.py.

Run:  python chat.py
Cmds: /sources  show sources of the last answer
      /clear    reset conversation memory
      /debug    toggle retrieval scores
      /exit     quit
"""
from rag import RagBot


def main() -> None:
    print("Loading...")
    try:
        bot = RagBot()
    except RuntimeError as e:
        raise SystemExit(str(e))
    print("Gnani.ai assistant ready. Commands: /sources /clear /debug /exit\n")

    debug, last_sources = False, []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/clear":
            bot.clear()
            print("bot> Conversation cleared.\n")
            continue
        if q == "/debug":
            debug = not debug
            print(f"bot> Debug {'on' if debug else 'off'}.\n")
            continue
        if q == "/sources":
            print("bot> " + ("\n     ".join(last_sources) or "No sources for the last answer.") + "\n")
            continue

        r = bot.ask(q)
        last_sources = r.sources
        if debug:
            print(f"  [debug] status={r.status} {r.detail}")
            for h in r.hits:
                print(f"  [debug] {h['sim']:.2f} {h['url']}")
        print(f"bot> {r.answer}")
        if r.sources:
            print("     Sources: " + ", ".join(r.sources[:3]))
        print()


if __name__ == "__main__":
    main()