.PHONY: setup data train train-baselines evaluate visualize tables all clean \
       train-clip evaluate-clip prepare-conversations train-world-model \
       evaluate-world-model all-stages

CONFIG ?= configs/default.yaml

setup:
	pip install -e .
	pip install -r requirements.txt

data:
	python scripts/prepare_data.py --config $(CONFIG)

train:
	python scripts/train.py --config $(CONFIG)

train-baselines:
	python scripts/train_baselines.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

visualize:
	python scripts/visualize.py --config $(CONFIG)

tables:
	python scripts/generate_tables.py --config $(CONFIG)

all: data train train-baselines evaluate visualize tables

# Stage B: CLIP-style intent alignment
train-clip:
	python scripts/train_clip.py --config $(CONFIG)

evaluate-clip:
	python scripts/evaluate_clip.py --config $(CONFIG)

# Stage C: World model
prepare-conversations:
	python scripts/prepare_conversations.py --config $(CONFIG)

train-world-model:
	python scripts/train_world_model.py --config $(CONFIG)

evaluate-world-model:
	python scripts/evaluate_world_model.py --config $(CONFIG)

# Full pipeline: all three stages
all-stages: data train train-baselines evaluate train-clip evaluate-clip \
            prepare-conversations train-world-model evaluate-world-model \
            visualize tables

clean:
	rm -rf data/processed checkpoints logs results
