from backend.models.schemas import CardSchema, GameStateRequest, PlayerState
from backend.services.game_service import get_game_service


def test_resolve_hand_result_rotates_positions_and_preserves_physical_seats():
    game_service = get_game_service()
    game_state = GameStateRequest(
        community_cards=[],
        pot=15,
        players=[
            PlayerState(
                position=0,
                stack=95,
                bet=0,
                hole_cards=[CardSchema(rank="A", suit="h"), CardSchema(rank="K", suit="s")],
                is_bot=True,
                is_active=False,
                has_acted=True,
            ),
            PlayerState(
                position=1,
                stack=90,
                bet=0,
                hole_cards=None,
                is_bot=False,
                is_active=False,
                has_acted=True,
            ),
            PlayerState(
                position=2,
                stack=100,
                bet=0,
                hole_cards=None,
                is_bot=False,
                is_active=True,
                has_acted=True,
            ),
        ],
        bot_position=0,
        seat_map=[3, 4, 5],
        starting_stacks=[100, 100, 100],
        current_bet=0,
        big_blind=10,
        current_player_idx=-1,
        street_raise_count=0,
        preflop_raise_count=0,
        preflop_call_count=0,
        preflop_last_raiser=None,
        last_aggressor=None,
        model_version="v24",
    )

    resolved = game_service.resolve_hand_result(game_state, [100, 100, 100], {})
    next_state = resolved["next_game_state"]

    assert next_state.seat_map == [4, 5, 3]
    assert next_state.bot_position == 2
    assert [player.position for player in next_state.players] == [0, 1, 2]
    assert [player.stack for player in next_state.players] == [90, 115, 95]
    assert next_state.seat_map[next_state.bot_position] == 3
