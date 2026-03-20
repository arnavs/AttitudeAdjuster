source /Users/arnavsood/MachineYearning/.venv/bin/activate && python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('submission/checkpoints/runs')
ea.Reload()
for tag in ['loss/value_betting_p0', 'loss/value_betting_p1', 'diag/loss_SB', 'diag/loss_BB', 'diag/regret_mag_SB', 'diag/regret_mag_BB', 'diag/fold_freq_SB', 'diag/fold_freq_BB']:
    events = ea.Scalars(tag)
    if events:
        print(f'{tag}:')
        for e in events:
            print(f'  iter {e.step:5d}: {e.value:.4f}')
        print()
"
