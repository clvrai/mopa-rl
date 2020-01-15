ffmpeg -f concat -safe 0 -i <(for f in ./logs/rl.reacher-obstacle-v0.baseline_v5.123/video/*.mp4; do echo "file '$PWD/$f'"; done) -c copy ./logs/rl.reacher-obstacle-v0.baseline_v5.123/video/output.mp4
ffmpeg -i ./logs/rl.reacher-obstacle-v0.baseline_v5.123/video/output.mp4 -vf "setpts=2*PTS" ./logs/rl.reacher-obstacle-v0.baseline_v5.123/video/output_slower.mp4
