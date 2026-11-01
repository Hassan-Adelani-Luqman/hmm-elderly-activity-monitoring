"""Build the five-page Word report from frozen Phase 7 artifacts and local links.

Run: python scripts/build_report.py [--video-links videos.txt]
Export with LibreOffice; page count/layout are verified separately on its PDF.
The local video note is not committed; evidence links belong in the report.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

ROOT = Path(__file__).resolve().parents[1]
NAVY = '17365D'


def hyperlink(paragraph, label, url):
    """Append an actual clickable external hyperlink to a Word paragraph."""
    element = OxmlElement('w:hyperlink')
    element.set(qn('r:id'), paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True))
    run = OxmlElement('w:r'); properties = OxmlElement('w:rPr')
    color = OxmlElement('w:color'); color.set(qn('w:val'), '0563C1'); properties.append(color)
    underline = OxmlElement('w:u'); underline.set(qn('w:val'), 'single'); properties.append(underline)
    run.append(properties)
    text = OxmlElement('w:t'); text.text = label; run.append(text); element.append(run)
    paragraph._p.append(element)


def table(document, caption, headers, rows, widths=None):
    """Add a numbered caption and a compact table with repeated, shaded headings."""
    paragraph(document, caption, style='Caption')
    result = document.add_table(rows=1, cols=len(headers))
    result.style = 'Table Grid'
    result.autofit = False
    if widths:
        for column, width in zip(result.columns, widths): column.width = Inches(width)
    for cell, text in zip(result.rows[0].cells, headers): cell.text = str(text)
    repeat = OxmlElement('w:tblHeader'); result.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in rows:
        for cell, text in zip(result.add_row().cells, row): cell.text = str(text)
    for i, row in enumerate(result.rows):
        avoid = OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(avoid)
        for j, cell in enumerate(row.cells):
            if widths: cell.width = Inches(widths[j])
            margins = OxmlElement('w:tcMar')
            for side in ['top', 'bottom']:
                item = OxmlElement('w:'+side); item.set(qn('w:w'), '35'); item.set(qn('w:type'), 'dxa'); margins.append(item)
            cell._tc.get_or_add_tcPr().append(margins)
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                for run in p.runs: run.font.size = Pt(9)
            if i == 0:
                shade = OxmlElement('w:shd'); shade.set(qn('w:fill'), 'E8EEF5'); cell._tc.get_or_add_tcPr().append(shade)
                for run in cell.paragraphs[0].runs: run.bold = True
    return result


def paragraph(document, text='', style=None):
    """Add report prose or a styled caption without manual line wrapping."""
    return document.add_paragraph(text, style=style)


def figure(document, filename, caption, width):
    """Embed the existing notebook-exported PNG and its numbered caption."""
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ROOT/'figures'/filename), width=Inches(width))
    paragraph(document, caption, style='Caption')


def build_report(video_links):
    """Generate DOCX using actual model metrics, counts, sampling rates and URLs."""
    config = json.loads((ROOT/'report/report_config.json').read_text())
    videos = {name.lower(): url for name, url in
              (line.split(': ', 1) for line in Path(video_links).read_text().splitlines() if line.strip())}
    assert set(videos) == {'still', 'standing', 'walking', 'jumping'}
    quality = json.loads((ROOT/'data/quality/status.json').read_text())
    model = json.loads((ROOT/'data/models/activity_models.json').read_text())
    metrics = pd.read_csv(ROOT/'data/evaluation/metrics.csv')
    comparison = pd.read_csv(ROOT/'data/evaluation/comparison.csv').set_index('model')
    clips = pd.read_csv(ROOT/'data/evaluation/per_clip.csv')
    metadata = pd.read_csv(ROOT/'data/metadata.csv')
    verification = json.loads((ROOT/'data/verification/hmm.json').read_text())
    primary = model['protocol']['primary_model']
    assert primary == 'baum_welch' and quality['accepted_clips'] == 52
    final_metrics = metrics.query("model == @primary and split == 'test'")
    em = model['models'][primary]
    assert int(comparison.loc[primary, 'test_windows']) == 65
    pct = lambda value: f'{value*100:.2f}%'
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.2677), Inches(11.6929)
    sec.top_margin = sec.bottom_margin = Inches(.6)
    sec.left_margin = sec.right_margin = Inches(.7)
    sec.header_distance = sec.footer_distance = Inches(.25)
    normal = doc.styles['Normal']; normal.font.name = 'Arial'; normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.03
    normal.paragraph_format.space_after = Pt(5)
    for name, size in [('Title', 18), ('Heading 1', 13), ('Heading 2', 11)]:
        style = doc.styles[name]; style.font.name = 'Arial'; style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(NAVY)
        style.paragraph_format.space_before = Pt(7 if name != 'Title' else 0)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True
    doc.styles['Caption'].font.name = 'Arial'
    doc.styles['Caption'].font.size = Pt(8.5)
    doc.styles['Caption'].font.bold = False
    doc.styles['Caption'].font.color.rgb = RGBColor.from_string('404040')
    doc.styles['Caption'].paragraph_format.space_after = Pt(5)
    header = sec.header.paragraphs[0]
    header.text = config['author'].upper()+'  |  HUMAN ACTIVITY HMM'
    header.runs[0].font.size = Pt(8)
    header.runs[0].font.color.rgb = RGBColor.from_string('666666')
    footer = sec.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('Page ').font.size = Pt(8)
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    footer.add_run(' of ').font.size = Pt(8)
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'NUMPAGES'); footer._p.append(field)
    doc.core_properties.author = config['author']
    doc.core_properties.title = config['title']
    doc.core_properties.subject = config['course']+' — '+config['assignment']
    doc.core_properties.keywords = 'HMM, activity recognition, accelerometer, gyroscope, Viterbi, Baum–Welch'

    # PAGE 1: context, collection, evidence and raw signals.
    doc.add_heading(config['title'], 0)
    p = paragraph(doc, f"{config['author']}  |  {config['course']}")
    p.runs[0].bold = True
    paragraph(doc, f"{config['assignment']}  •  {config['date']}")
    p = paragraph(doc, 'Repository: '); hyperlink(p, config['repository'], config['repository'])
    doc.add_heading('Background and Motivation', 1)
    paragraph(doc, 'Falls and long, unnoticed periods of inactivity are serious risks for older adults who live alone, and a phone they already carry could watch for both. My use case is a monitor that recognises everyday activity states from the phone’s accelerometer and gyroscope and flags unusual patterns, such as a sudden impact followed by a long stillness, for a carer to check. The activity is never measured directly, only noisy sensor signals are, so I model it as a hidden state sequence: a Hidden Markov Model over four states (still, standing, walking and jumping). Jumping stands in for a sudden impact, because real falls cannot be recorded safely.')
    doc.add_heading('Data Collection and Pre-processing', 1)
    paragraph(doc, 'I recorded all data myself with Sensor Logger 1.68 on an Infinix X6855 (Android 36): 52 single-activity clips, 13 per activity, indoors on 25, 27 and 28 October 2026. For standing, walking and jumping I held the phone at waist level, screen facing me and top edge up; for still, the phone lay flat on a table. Each clip was cropped to its steady part (5–10 s), giving at least 90 s per activity (Table 1). I used TotalAcceleration (m/s², with gravity) rather than the gravity-free stream, because gravity’s direction shows how the phone is held.')
    clip_meta = metadata[metadata.type=='clip']
    table(doc, 'Table 1. Collection summary and sampling rates (single participant, one phone).',
          ['Recordings', 'Files / retained seconds', 'App setting → measured → harmonised'],
          [['Activity clips', '52; still 130.00, standing 119.84,\nwalking 112.72, jumping 99.92 s',
            f"100 → {clip_meta.measured_acc_hz.min():.2f}–{clip_meta.measured_acc_hz.max():.2f} → 25 Hz"],
           ['Lower-rate pilot', '1 recording (phone flat)', '50 → 29.23 → 25 Hz']], [1.05,2.9,2.9])
    p = paragraph(doc, 'Data-collection videos: ')
    for index, activity in enumerate(['still','standing','walking','jumping']):
        if index: p.add_run('  |  ')
        hyperlink(p, activity.title(), videos[activity])
    figure(doc, 'fig1_raw_signals.png', 'Figure 1. Raw |a| (blue, left axis) and |ω| (orange, right axis) for one clip per activity, before filtering. Axis ranges differ by activity: still varies by about 0.02 m/s², standing shows small sway, walking a regular step rhythm, and jumping large periodic impacts near 40 m/s².', 4.1)

    # PAGE 2: sampling-rate harmonisation, windows, features and the HMM.
    doc.add_page_break()
    doc.add_heading('Sampling rates, cleaning and windows', 2)
    paragraph(doc, 'A pilot recording showed that the phone does not deliver the rate set in the app: the 100 Hz setting gives 58.4 Hz and the 50 Hz setting gives 29.2 Hz. The accelerometer and gyroscope are also sampled at different instants and never share a timestamp. I therefore harmonise every recording the same way: a fourth-order zero-phase Butterworth low-pass filter at 10 Hz, then one linear interpolation of both streams onto a common 25 Hz grid. I chose 25 Hz because it is below every delivered rate, so no recording is ever upsampled, and its 12.5 Hz Nyquist limit still covers the walking and jumping rhythms (about 1.5–2.5 Hz) and their main harmonics.')
    paragraph(doc, 'My two planned continuous sessions (about two minutes each, covering all four activities) suffered sensor outages, including an 87.5 s gap in session 2, and I had no exact record of their activity boundaries. I did not fill or bridge the gaps: I recovered four unbroken blocks without labels, kept them out of training, and use them only to inspect decoded transitions (Figure 4).')
    paragraph(doc, 'Windows are 2 s long (50 samples at 25 Hz) with a 1 s hop. Two seconds holds three or more step or jump cycles and gives 0.5 Hz FFT resolution; because the window is defined in seconds, it covers the same time span whatever a recording’s native rate. Each window takes the majority label of its samples, and windows never cross file boundaries, giving 338 training and 65 test windows.')
    table(doc, 'Table 2. The ten retained features and why each helps separate the activities.',
          ['Features', 'Domain', 'Justification'],
          [['Mean acc_x; mean acc_y', 'Time (2)', 'Gravity direction: flat phone (still) versus upright phone (other activities)'],
           ['Std of |a|', 'Time', 'Motion intensity: still < standing < walking < jumping'],
           ['Variance of |ω|', 'Time', 'How much the phone rotates while moving'],
           ['SMA acceleration; SMA gyro', 'Time (2)', 'Overall activity level and rotation (mean sum of absolute axes)'],
           ['Correlation acc_y, acc_z', 'Time', 'Coordinated movement between axes versus uncorrelated noise'],
           ['Dominant frequency of |a|', 'FFT', 'Step or jump rate; no clear peak when still or standing'],
           ['Band-energy ratio of |a|', 'FFT', 'Low-band gait energy [0.5,3] Hz versus higher band (3,10] Hz'],
           ['Spectral entropy of |a|', 'FFT', 'Periodic motion has low entropy; sensor noise has high entropy']], [2.15,.7,4.0])
    paragraph(doc, 'I computed 14 candidate features and dropped four (mean acc_z, centred RMS, range and spectral energy) that correlated above |r| = 0.95 with a retained feature in the training data, leaving 7 time-domain and 3 FFT features. FFT features use the demeaned |a|, a Hann window and one-sided power, ignoring the DC bin. All features are Z-score normalised with means and standard deviations from the 44 training clips only. Z-scores suit Gaussian emissions: the raw features mix m/s², rad/s, Hz and unitless ratios, and without scaling the largest-unit features would dominate every likelihood. Fitting on training data only keeps test information out of the model.')
    doc.add_heading('HMM Setup and Implementation Details', 1)
    table(doc, 'Table 3. HMM components (implemented from scratch in NumPy).', ['Element', 'Definition'],
          [['Z', 'Four hidden states: still, standing, walking, jumping'],
           ['X', 'One ten-dimensional, Z-scored feature vector per 2 s window'],
           ['A', '4×4 transition matrix: A[i,j] = P(next state j | current state i)'],
           ['B', 'Gaussian emission density per state, diagonal covariance, variance floor 1e-3'],
           ['π', 'Initial-state probabilities; one sequence start per recording']], [.85,6.0])
    paragraph(doc, 'Three algorithms are needed. The forward–backward algorithm, in scaled log space to avoid underflow, gives the sequence likelihood and the state posteriors. Baum–Welch (EM) uses those posteriors to re-estimate the parameters and stops when the total log-likelihood improves by less than 1e-4; 500 iterations is only a safety cap. Viterbi then finds the single most probable state path. Sensitivity, specificity and accuracy are computed from one hard label per window, so Viterbi’s joint path is the decoder that matches the metrics, and it uses A to keep neighbouring labels consistent.')
    paragraph(doc, 'Training starts from supervised estimates: per-activity Gaussian means and variances, and transition counts with one Laplace pseudocount per entry. Every training clip contains one activity, so the counts contain no switches. In my first version, free Baum–Welch therefore drove every unobserved transition to zero and made still and jumping absorbing states. My primary model keeps the supervised transition prior fixed and lets EM re-estimate π and the emissions; the free-A model is kept for comparison. I verified the code against exact ground truth: enumerating all 64 paths of a two-state example reproduces the likelihood, Viterbi path, posteriors and one EM update, and a known three-state model is recovered from synthetic data in 7 iterations. All 65 regression tests pass.')

    # PAGE 3: parameters and the rubric tables.
    doc.add_page_break()
    figure(doc, 'fig2_transition_emissions.png', 'Figure 2. Primary model parameters. (a) Transition matrix A: the supervised persistence prior, kept fixed during EM; rows sum to one. (b) Gaussian emission means in training z-scores; each state also has a diagonal variance.', 5.25)
    diagonal = np.diag(np.array(em['parameters']['A']))
    paragraph(doc, f'Each activity stays the same from one 1 s step to the next with probability {diagonal.min():.3f}–{diagonal.max():.3f}, with a small equal chance of switching to any other activity. That captures one realistic property, that activities last many seconds, but not which switches are realistic: still→jumping should be far rarer than still→standing, and learning that needs labelled continuous recordings. The emission means show why the classes separate. Mean acc_y (orientation) puts still far from the three upright activities, and every motion-intensity feature puts jumping far from the rest. Standing and walking differ mainly in small motion features, which is where the errors occur.')
    doc.add_heading('Results and Interpretation', 1)
    paragraph(doc, 'Before fitting anything, I held out two whole clips per activity (8 clips, 65 windows), chosen with a fixed random seed, and never used them for fitting, feature selection, scaling or state mapping. They come from the same days and conditions as training, so they test recognition on new recordings rather than in a new environment. Each test clip is decoded on its own.')
    rows=[]
    for name, label in [('supervised','Supervised HMM'), ('baum_welch','Baum–Welch, A fixed (primary)'),
                        ('baum_welch_free_transitions','Baum–Welch, free A'), ('gaussian_baseline','Independent Gaussian (no A)')]:
        row=comparison.loc[name]
        rows.append([label, f"{int(row.train_correct)}/{int(row.train_windows)} ({pct(row.train_accuracy)})", f"{int(row.test_correct)}/{int(row.test_windows)} ({pct(row.test_accuracy)})"])
    table(doc, 'Table 4. Model comparison (training accuracy is measured on the training data itself).', ['Model', 'Training correct / windows', 'Test correct / windows'], rows, [2.55,2.15,2.15])
    table(doc, 'Table 5. Primary model on the held-out data; samples are windows.',
          ['State (Activity)', 'Number of\nSamples', 'Sensitivity', 'Specificity', 'Overall\nAccuracy*'],
          [[row.activity.title(), int(row.n_windows), pct(row.sensitivity), pct(row.specificity), pct(row.accuracy)] for row in final_metrics.itertuples()],
          [1.55,1.0,1.4,1.4,1.5])
    correct = int(comparison.loc[primary, 'test_correct'])
    paragraph(doc, f'*Sensitivity = TP/(TP+FN); specificity = TN/(TN+FP). Activity rows give one-vs-rest accuracy (TP+TN)/65, which true negatives inflate; the Overall row gives macro sensitivity and specificity and the true multiclass accuracy ({correct}/65).', style='Caption')

    # PAGE 4: confusion, convergence and per-clip outcomes.
    doc.add_page_break()
    figure(doc, 'fig3_confusion_convergence.png', 'Figure 3. (a) Held-out confusion matrix for the primary model: two standing windows are predicted as walking. (b) Training log-likelihood during Baum–Welch, which stops once the improvement falls below 1e-4.', 6.55)
    paragraph(doc, f"Baum–Welch raised the training log-likelihood from {em['history'][0]:.2f} to {em['history'][-1]:.2f} in {em['iterations']} iterations, yet both EM models classify {correct}/65 held-out windows ({pct(comparison.loc[primary,'test_accuracy'])}), one fewer than the supervised HMM and the baseline ({pct(comparison.loc['supervised','test_accuracy'])}). A better fit to the training data did not mean better classification. On single-activity clips the transition matrix has nothing to add, so fixing A changed no held-out prediction.")
    test_clips = clips.query("model == @primary and split == 'test'").set_index('sequence_id')
    clip_rows=[]
    for activity in ['still','standing','walking','jumping']:
        for name,row in test_clips[test_clips.activity==activity].sort_index().iterrows():
            clip_rows.append([name, int(row.n_windows), int(row.correct), int(row.errors), pct(row.accuracy)])
    table(doc, 'Table 6. Primary model results for each of the eight held-out clips.',
          ['Recording', 'Windows', 'Correct', 'Errors', 'Accuracy'], clip_rows, [2.45,1.1,1.1,1.1,1.1])
    paragraph(doc, 'Both errors are in standing_06 (windows 3 and 8). Window 3 coincides with a brief burst of movement in the signal, and window 8 is the clip’s last window, with no later evidence to pull the decoder back to standing. The other seven clips are decoded without error, so standing sensitivity is 88.89% and walking specificity 95.74%, while still and jumping are perfect. With only two test clips per activity and overlapping windows, these figures are encouraging but not precise.')
    paragraph(doc, 'The independent Gaussian baseline classifies each window from the supervised emissions alone, ignoring A and neighbouring windows. It matches the supervised HMM here because single-activity clips leave nothing for the transitions to correct; the value of A only shows on continuous data (Figure 4). Re-running the notebook reproduces every fitted parameter and result exactly.')

    # PAGE 5: continuous decoding, discussion and conclusion.
    doc.add_page_break()
    figure(doc, 'fig4_session_decoding.png', 'Figure 4. Viterbi decoding of the unlabelled continuous session 1 (three blocks; hatched gaps are sensor outages, never joined). The planned schedule is my approximate timer protocol, not verified truth. Bottom: processed |a|.', 6.45)
    doc.add_heading('Discussion and Conclusion', 1)
    paragraph(doc, 'Figure 4 is the most revealing result. With free transitions, the model labels the first two blocks entirely as jumping: once in an absorbing state it can never leave, so it would be useless as a continuous monitor. The primary model does switch. It finds the still periods at the start and end and the real jumping burst, but it decodes most of the session’s walking and standing as jumping, because I held the phone differently in that session. Its mean acc_y sits near −0.6 z, against +0.65 z for the standing and walking clips, and the broad jumping emission absorbs these unfamiliar windows. The model generalises well to new clips recorded the same way, but not to a different grip.')
    paragraph(doc, 'Still and jumping were the easiest activities: orientation alone separates still, and jumping’s impacts dominate every intensity feature. Standing versus walking was the hardest, because with the phone held steady at the waist, walking adds only a little extra motion. The pair I expected to be hardest, still versus standing, was solved by the direction of gravity.')
    paragraph(doc, 'The delivered sampling rate (58.4 Hz) differed from the app setting (100 Hz), and the two sensors never shared a timestamp, so harmonisation was essential rather than optional. Filtering at 10 Hz removes noise and aliasing but also rounds off jump impacts; that did not hurt jumping here, but it would matter for short, sharp falls. The session outages were my largest data problem and cost me labelled transition data.')
    paragraph(doc, 'The biggest improvements would be labelled continuous recordings, to learn realistic transitions and test real switching; a fixed, realistic phone placement (pocket or belt) or orientation-invariant features, so that a change of grip does not break the model; more participants and devices; and extra sensors such as a barometer for falls. A duration-aware model (a hidden semi-Markov model) would also suit activities that last many seconds.')
    paragraph(doc, f"In conclusion, my from-scratch HMM is verified against exact results and recognises held-out single-activity clips well ({pct(comparison.loc[primary,'test_accuracy'])}, with still and jumping perfect). Building it taught me that the weak points of activity recognition lie in the data rather than the algorithm: the transition matrix is only as good as the transition data, and features tied to phone orientation make the model sensitive to how the phone is held.")
    # Put page breaks on the following content, so a trailing break paragraph
    # cannot spill onto an otherwise blank extra page in Word/LibreOffice.
    for p in list(doc.paragraphs):
        if p._p.xpath('.//w:br[@w:type="page"]'):
            following = p._p.getnext()
            flag = OxmlElement('w:pageBreakBefore')
            following.get_or_add_pPr().append(flag)
            p._p.getparent().remove(p._p)
    output = ROOT/'report/HMM_Activity_Recognition_Report.docx'
    doc.save(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-links', type=Path, default=ROOT/'videos.txt')
    print(build_report(parser.parse_args().video_links))
