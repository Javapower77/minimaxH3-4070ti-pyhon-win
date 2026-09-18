from minimax_h3_fl2v.prompts import expand_prompt


def test_wraps_plain_text():
    out = expand_prompt("A red bicycle rolls through fog.")
    assert out.startswith("integrated_multimodal_description:")
    assert "overall_soundscape:" in out
    assert "non_diegetic_music:" in out


def test_keeps_structured_text():
    src = "integrated_multimodal_description: already structured\noverall_soundscape: rain\nnon_diegetic_music: none"
    assert expand_prompt(src) == src
