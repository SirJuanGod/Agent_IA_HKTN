import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rbj.language import LanguageModule
from rbj.memory import MemoryModule


CHECKPOINT = (
    "checkpoints/sjg_language_037.pt"
)


def main():

    print("=" * 70)
    print("SJG-AGENT — LANGUAGE + MEMORY")
    print("=" * 70)

    language = LanguageModule(
        checkpoint_path=CHECKPOINT,
        threshold=0.70,
    )

    memory = MemoryModule()

    commands = [
        "ve hacia la derecha",
        "avanza tres casillas hacia la izquierda",
        "detente",
    ]

    for text in commands:

        print("\n" + "-" * 70)
        print("INPUT:")
        print(text)

        result = language.process(
            text
        )

        print("\nLANGUAGE RESULT:")
        print(result["status"])

        if result["status"] != "READY":

            print(
                "Reason:",
                result.get("reason"),
            )

            continue

        command = result["command"]

        # ---------------------------------------------
        # STORE CURRENT COMMAND
        # ---------------------------------------------

        memory.set_command(
            command
        )

        # ---------------------------------------------
        # STORE HISTORY
        # ---------------------------------------------

        memory.add_history({
            "event": "COMMAND_RECEIVED",
            "text": text,
            "command": command,
        })

        print("\nCOMMAND STORED:")
        print(
            memory.get_command()
        )

    # =================================================
    # FINAL MEMORY
    # =================================================

    print("\n" + "=" * 70)
    print("FINAL MEMORY")
    print("=" * 70)

    print(
        memory.summary()
    )

    print("\nHISTORY:")

    for event in memory.history():

        print(event)


if __name__ == "__main__":
    main()