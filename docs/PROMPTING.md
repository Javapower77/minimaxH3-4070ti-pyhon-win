# Prompting MiniMax-H3 FL2VA

MiniMax-H3 is trained on **H3-Context-IR**: a structured description of
picture, camera, motion, diegetic sound, and score. It is not a chat
model. Keep prompts **specific and compact**.

## Template

```
integrated_multimodal_description: <what we see, camera, lighting, motion>
overall_soundscape: <diegetic audio>
non_diegetic_music: <score, or "none">
```

The Gradio checkbox wraps a plain sentence into this template. If those
three headings are already present, the text is used unchanged.

## FL2VA-specific advice

- Describe the **path from first frame to last frame** (walk, turn, weather).
- First and last stills should share identity, wardrobe, and lens.
- Extreme pose jumps between stills cause morphing.
- Mention audio even if you care about picture; the audio VAE always runs.
- No CFG — you cannot "negative prompt" the usual way. Say what you want.

## Good prompt (short)

```
integrated_multimodal_description: A woman in a rust-red coat walks through
neon rain toward a subway entrance. Handheld, 35 mm, shallow DOF, cyan and
sodium highlights. She stops under the station sign in the last frame and
looks back.

overall_soundscape: Rain hiss, heels on wet concrete, distant traffic,
subway rumble, no speech.

non_diegetic_music: Low analog pad, quiet, never over the rain.
```

## Weak prompt

```
make a cinematic video of a woman, 8k, masterpiece, best quality
```

That style wastes encoder tokens and does not describe motion or audio.

## Duration

5 s (124 frames) is the quality/speed sweet spot on one H100. 10–15 s
works but costs linear time and more offload traffic.
