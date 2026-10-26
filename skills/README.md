# MLX skills

This repo ships two skills:

- `skills/mlx-swift/` — MLX Swift arrays, neural networks, autodiff, and related
  low-level APIs.
- `skills/mlxkit/` — using the MLXKit training workflow from consumer projects.

In the commands below, replace `<skill>` with `mlx-swift` or `mlxkit`.

## Install globally (home directory)

Run these from the MLXKit repo root, or replace `$(pwd)` with its absolute path.

### Claude Code

```sh
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/<skill>" ~/.claude/skills/<skill>
```

### Codex

```sh
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/<skill>" ~/.codex/skills/<skill>
```

### Droid

```sh
mkdir -p ~/.agents/skills
ln -s "$(pwd)/skills/<skill>" ~/.agents/skills/<skill>
```

## Install per-project

Create a local skills folder in the consumer project. Set `MLXKIT_REPO` to the
absolute path of this checkout, then run the commands from the consumer project
root.

### Claude Code

```sh
mkdir -p .claude/skills
ln -s "$MLXKIT_REPO/skills/<skill>" .claude/skills/<skill>
```

### Codex

```sh
mkdir -p .codex/skills
ln -s "$MLXKIT_REPO/skills/<skill>" .codex/skills/<skill>
```

### Droid

```sh
mkdir -p .agents/skills
ln -s "$MLXKIT_REPO/skills/<skill>" .agents/skills/<skill>
```

## Notes

- If your tool caches skills, restart it after installing.
- If you prefer a copy over a symlink, replace `ln -s` with `cp -R`.
