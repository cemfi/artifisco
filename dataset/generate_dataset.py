import torchaudio
import torch
from PIL import Image
import os
import random
import shutil
import subprocess
import tempfile
from glob import glob
from multiprocessing.pool import Pool

import lxml.etree as ET
from torchvision import transforms
from tqdm import tqdm

N_FFT = 2048
SNIPPET_LENGTH_IN_SECONDS = 15

def process(mei_filepath):
    mei_filepath = os.path.abspath(mei_filepath)
    basename = os.path.basename(os.path.splitext(mei_filepath)[0])
    output_dir = os.path.join('.', 'data')
    os.makedirs(output_dir, exist_ok=True)
    dir = os.path.join('.', 'raw', basename)
    new_mei_filepath = os.path.join(dir, 'meico.mei')
    # os.makedirs(dir)
    #
    # os.chdir(dir)
    # subprocess.call([
    #     '../../utils/verovio',
    #     '--resources', '../../utils/verovio-data',
    #     '--all-pages',
    #     '--scale', '60',
    #     '--header', 'none',
    #     '--footer', 'none',
    #     '--font', random.choice(['Bravura', 'Leipzig', 'Gootville', 'Petaluma']),
    #     '--outfile', 'page',  # use prefix 'page_'
    #     mei_filepath,
    # ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #
    # os.chdir('../..')
    # shutil.copy(mei_filepath, new_mei_filepath)
    # subprocess.call([
    #     'java',
    #     '-Xmx16g',
    #     '-jar', './utils/meica.jar',
    #     new_mei_filepath,
    #     str(SNIPPET_LENGTH_IN_SECONDS * 2)
    # ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ids_wav = ['_'.join(os.path.splitext(os.path.basename(m))[0].split('_')[:-1]) for m in glob(f"{dir}/*.wav")]

    to_tensor = transforms.ToTensor()
    to_stft = torchaudio.transforms.Spectrogram(n_fft=N_FFT, power=2)

    with tempfile.TemporaryDirectory() as temp_dir:
        for p, svg_filepath in enumerate(sorted([d for d in glob(os.path.join(dir, '*.svg')) if '-new.svg' not in d])):
            svg_basename = os.path.splitext(os.path.basename(svg_filepath))[0]

            svg_filepath_new = os.path.join(temp_dir, f'{svg_basename}-new.svg')
            svg_root = ET.parse(svg_filepath).getroot()

            # Modify SVG:
            # - Prepend "notehead-" to notehead elements
            # - Remove barlines
            for e, elem in enumerate(
                    svg_root.xpath('//*[local-name()="use" and (@href="#E0A4" or @href="#E0A3" or @href="#E0A2")]')):
                elem.set('id', f'notehead-{e}')
            for elem in svg_root.xpath('//*[@class="barLineAttr"]'):
                elem.getparent().remove(elem)

            svg_string = ET.tostring(svg_root, pretty_print=True)
            with open(svg_filepath_new, 'wb') as f:
                f.write(svg_string)

            svg_height = int(svg_root.get('height').replace('px', ''))

            # Get bboxes for all elements from Inkscape
            bboxes = subprocess.check_output([
                'inkscape',
                '--without-gui',
                '--query-all',
                svg_filepath_new
            ]).decode('utf-8')

            # Format bboxes
            bboxes = [bbox.split(',') for bbox in bboxes.splitlines()]
            bboxes_dict = {}
            for bbox in bboxes:
                bboxes_dict[bbox[0]] = {
                    'x': int(float(bbox[1])),
                    'y': int(float(bbox[2])),
                    'w': int(float(bbox[3])),
                    'h': int(float(bbox[4]))
                }

            # Generate data per measure
            for m, measure in enumerate(svg_root.xpath('//*[@class="measure"]')):
                measure_id = measure.get('id')
                # Skip measure that have no audio part for whatever reason
                if measure_id not in ids_wav:
                    continue

                staff_bbox = bboxes_dict[measure.xpath('.//*[@class="staff"]')[0].get('id')]
                measure_bbox = bboxes_dict[measure_id]

                measure_l = staff_bbox['x']
                measure_r = staff_bbox['x'] + staff_bbox['w']
                measure_b = svg_height - measure_bbox['y']
                measure_t = measure_b - measure_bbox['h']

                # Get measure area ####################################################
                target_filename = os.path.join(temp_dir, 'measure.png')
                # target_filename = os.path.join(dir, f"{measure_id}.png")
                subprocess.call([
                    'inkscape',
                    f'--export-area={measure_l}:{measure_b}:{measure_r}:{measure_t}',
                    '--export-background=white',
                    f'--export-png={target_filename}',
                    svg_filepath_new
                ], stdout=subprocess.DEVNULL)

                # Convert to Greyscale
                image = Image.open(target_filename)
                image = image.convert('L')
                image = image.resize((512, 512))
                image_tensor = to_tensor(image).squeeze(dim=0)
                image.close()
                del image

                # Process audio #######################################################
                wav_filepath = glob(f"{dir}/{measure_id}_*.wav")[0]
                wave, _ = torchaudio.load_wav(wav_filepath)
                wave = wave[0, :]  # Left channel only

                # Get to correct snippet length
                maximum_length = 44100 * SNIPPET_LENGTH_IN_SECONDS
                if wave.shape[0] > maximum_length:
                    wave = wave[:maximum_length]
                else:
                    missing_length = maximum_length - wave.shape[0]
                    wave = torch.cat((
                        wave,
                        torch.zeros(missing_length)
                    ), 0)

                stft = to_stft.forward(wave.unsqueeze(dim=0))
                spectrum = torch.log(1 + stft.squeeze(dim=0))

                target_milliseconds = float("_".join(os.path.splitext(os.path.basename(wav_filepath))[0].split('_')[1:]))
                target_samples = target_milliseconds * 44.1
                target_window = int(target_samples / N_FFT * 2)

                # Skip if measure was actually longer than the snippet length
                if target_window >= stft.shape[2]:
                    continue

                # import matplotlib.pyplot as plt
                # plt.subplot(1, 2, 1)
                # plt.imshow(spectrum[:300,:], origin='lower')
                # plt.axvline(target_window, color='red', linewidth=1)
                # plt.subplot(1, 2, 2)
                # plt.imshow(image_tensor[0], cmap='Greys_r')
                # plt.tight_layout()
                # figManager = plt.get_current_fig_manager()
                # figManager.window.showMaximized()
                # plt.show()

                # Save everything #####################################################
                torch.save({
                    'image': image_tensor,
                    'spectrum': spectrum,
                    'target': target_window
                }, os.path.join(output_dir, f"{basename}_{measure.get('id')}.pth"))



if __name__ == '__main__':
    pool = Pool(max(1, os.cpu_count() - 4))  # Leave some cpus for other things...
    mei_filepaths = sorted(glob('./mei/*.mei'))
    for _ in tqdm(pool.imap_unordered(process, mei_filepaths), total=len(mei_filepaths), unit='MEI'):
        pass
    # process(sorted(glob('./mei/*.mei'))[2])
