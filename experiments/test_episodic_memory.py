import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rbj.memory import EpisodicMemory


def main():

    memory = EpisodicMemory(
        capacity=10
    )

    print("=" * 70)
    print("SJG-AGENT — EPISODIC MEMORY TEST")
    print("=" * 70)

    # -------------------------------------------------
    # EPISODIO 1
    # -------------------------------------------------

    episode_1 = memory.add_episode(

        state_before={
            "robot_position": (2, 2),
            "robot_status": "IDLE",
        },

        command={
            "action": "MOVE",
            "direction": "RIGHT",
            "distance": 1,
        },

        action={
            "action": "MOVE",
            "direction": "RIGHT",
            "distance": 1,
        },

        state_after={
            "robot_position": (3, 2),
            "robot_status": "MOVING",
        },

        reward=1.0,

        success=True,

        result="ACTION_COMPLETED",
    )

    # -------------------------------------------------
    # EPISODIO 2
    # -------------------------------------------------

    episode_2 = memory.add_episode(

        state_before={
            "robot_position": (3, 2),
            "robot_status": "IDLE",
        },

        command={
            "action": "MOVE",
            "direction": "UP",
            "distance": 2,
        },

        action={
            "action": "MOVE",
            "direction": "UP",
            "distance": 2,
        },

        state_after={
            "robot_position": (3, 4),
            "robot_status": "MOVING",
        },

        reward=-0.5,

        success=False,

        result="COLLISION",
    )

    # -------------------------------------------------
    # EPISODIO 3
    # -------------------------------------------------

    memory.add_episode(

        state_before={
            "robot_position": (3, 2),
            "robot_status": "IDLE",
        },

        command={
            "action": "GO_TO",
            "target": "GOAL",
        },

        action={
            "action": "GO_TO",
            "target": "GOAL",
        },

        state_after={
            "robot_position": (4, 4),
            "robot_status": "GOAL_REACHED",
        },

        reward=10.0,

        success=True,

        result="GOAL_REACHED",
    )

    # -------------------------------------------------
    # MOSTRAR EPISODIOS
    # -------------------------------------------------

    print("\nEPISODIOS:")

    for episode in memory.all():

        print(
            f"\nEpisode {episode.episode_id}"
        )

        print(
            "Command:",
            episode.command
        )

        print(
            "Action:",
            episode.action
        )

        print(
            "Reward:",
            episode.reward
        )

        print(
            "Success:",
            episode.success
        )

        print(
            "Result:",
            episode.result
        )

    # -------------------------------------------------
    # RECIENTES
    # -------------------------------------------------

    print("\nEPISODIOS RECIENTES:")

    for episode in memory.recent(2):

        print(
            episode.episode_id,
            episode.result,
        )

    # -------------------------------------------------
    # SUCCESS
    # -------------------------------------------------

    print("\nEPISODIOS EXITOSOS:")

    for episode in memory.successful():

        print(
            episode.episode_id,
            episode.result,
        )

    # -------------------------------------------------
    # FAILED
    # -------------------------------------------------

    print("\nEPISODIOS FALLIDOS:")

    for episode in memory.failed():

        print(
            episode.episode_id,
            episode.result,
        )

    # -------------------------------------------------
    # BY ACTION
    # -------------------------------------------------

    print("\nEPISODIOS MOVE:")

    for episode in memory.by_action(
        "MOVE"
    ):

        print(
            episode.episode_id,
            episode.action,
        )

    # -------------------------------------------------
    # STATISTICS
    # -------------------------------------------------

    print("\nESTADÍSTICAS:")

    print(
        "Total:",
        memory.count()
    )

    print(
        "Success rate:",
        memory.success_rate()
    )

    print(
        "Average reward:",
        memory.average_reward()
    )

    print("\nSUMMARY:")

    print(
        memory.summary()
    )


if __name__ == "__main__":
    main()