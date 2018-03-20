from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Performance plot
for scale in [3, 5]:
    if scale == 3:
        test_set = 'Test_SSH'
    elif scale == 5:
        test_set = 'Test_SST'
    print(test_set)
    time = []
    psnr = []
    model = []
    print(sorted(Path('.').glob(f'*-sc{scale}')))
    for save_dir in sorted(Path('.').glob(f'*-sc{scale}')):
        if 'bicubic' not in save_dir.stem:
            model += [save_dir.stem.rsplit('-', 1)[0].upper()]
            metrics_file = save_dir / f'test/{test_set}/metrics.csv'
            metrics = pd.read_csv(str(metrics_file), index_col='name')
            time += [metrics.time.average]
            psnr += [metrics.psnr.average]
    plt.figure()
    plt.semilogx(time, psnr, '.')
    plt.grid(True, which='both')
    for x, y, s in zip(time, psnr, model):
        if 'NS' in s:
            s = s.split('-')[1]
        plt.text(x, y, s)
    plt.xlabel('Run time (sec)')
    plt.ylabel('PSNR (dB)')
    plt.savefig(str(results_dir / f'performance-sc{scale}-{test_set}.png'))
    plt.close()

# History plot
for scale in [3, 5]:
    if scale == 3:
        bicub_score = 83
    elif scale ==5:
        bicub_Score = 81
    plt.figure()
    for save_dir in sorted(Path('.').glob(f'*-sc{scale}')):
        if 'bicubic' not in save_dir.stem:
            model = save_dir.stem.rsplit('-', 1)[0].upper()
            history_file = save_dir / f'train/history.csv'
            history = pd.read_csv(str(history_file))
            plt.plot(history.epoch, history.val_psnr, label=model, alpha=0.8)
    plt.axhline(bicub_score,color='k',linestyle='--')
    plt.legend()
    plt.xlim(0, 500)
    plt.xlabel('Epochs')
    plt.ylabel('Average test PSNR (dB)')
    plt.savefig(str(results_dir / f'history-sc{scale}.png'))

