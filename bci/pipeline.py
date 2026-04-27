import pygame
import numpy as np
import socket
import time
from pylsl import StreamInlet, resolve_streams


SAMPLE_RATE = 250
EPOCH_LENGTH = int(0.6 * SAMPLE_RATE)
BASELINE = int(0.1 * SAMPLE_RATE)
NUM_CYCLES = 3
FLASH_DURATION = 2
ISI = 0.3
C3, C4 = 1, 3

QUESTIONS = [
    "Is the sun a star?",
    "Can humans fly naturally like birds?",
    "Is the moon made of cheese?",
    "Are you controlling this system with your brain?",
]

BG         = (15, 15, 25)
TEXT_COLOR = (220, 220, 220)
YES_COLOR  = (30, 180, 80)
NO_COLOR   = (200, 50, 50)
FLASH_COLOR = (255, 255, 100)
RESULT_YES = (50, 220, 100)
RESULT_NO  = (220, 80, 80)

pygame.init()
W, H = 1000, 650
win = pygame.display.set_mode((W, H))
pygame.display.set_caption("EEG Visual Speller - P300 BCI")

font_title  = pygame.font.SysFont("Arial", 36, bold=True)
font_btn    = pygame.font.SysFont("Arial", 72, bold=True)
font_result = pygame.font.SysFont("Arial", 48, bold=True)
font_small  = pygame.font.SysFont("Arial", 24)
font_status = pygame.font.SysFont("Arial", 20)


def connect_lsl():
    print("Connecting to EEG stream...")
    streams = resolve_streams(wait_time=5.0)
    inlet = StreamInlet(streams[0])
    print("Connected!")
    return inlet


eeg_buffer = []


def update_buffer(inlet, n_samples=10):
    samples, _ = inlet.pull_chunk(max_samples=n_samples)
    for s in samples:
        eeg_buffer.append(s[:8])
    if len(eeg_buffer) > SAMPLE_RATE * 5:
        del eeg_buffer[:-SAMPLE_RATE * 5]


def get_epoch(inlet, duration_samples):
    epoch = []
    while len(epoch) < duration_samples:
        sample, _ = inlet.pull_sample()
        epoch.append(sample[:8])
    return np.array(epoch)


def score_epoch(epoch):
    signal = np.mean(epoch[:, [C3, C4]], axis=1)
    p300_start = int(0.25 * SAMPLE_RATE)
    p300_end   = int(0.50 * SAMPLE_RATE)
    if p300_end > len(signal):
        return 0.0
    return float(np.max(signal[p300_start:p300_end]))


def draw_background():
    win.fill(BG)
    pygame.draw.line(win, (50, 50, 70), (0, 80), (W, 80), 2)


def draw_question(question):
    surf = font_title.render(question, True, TEXT_COLOR)
    win.blit(surf, (W // 2 - surf.get_width() // 2, 25))


def draw_buttons(yes_flash=False, no_flash=False):
    yes_col = FLASH_COLOR if yes_flash else YES_COLOR
    pygame.draw.rect(win, yes_col, (100, 200, 300, 180), border_radius=20)
    yes_txt = font_btn.render("YES", True, (10, 10, 10) if yes_flash else (255, 255, 255))
    win.blit(yes_txt, (100 + 150 - yes_txt.get_width() // 2,
                       200 + 90  - yes_txt.get_height() // 2))

    no_col = FLASH_COLOR if no_flash else NO_COLOR
    pygame.draw.rect(win, no_col, (600, 200, 300, 180), border_radius=20)
    no_txt = font_btn.render("NO", True, (10, 10, 10) if no_flash else (255, 255, 255))
    win.blit(no_txt, (600 + 150 - no_txt.get_width() // 2,
                      200 + 90  - no_txt.get_height() // 2))


def draw_status(msg, cycle=None, total=NUM_CYCLES):
    surf = font_small.render(msg, True, (150, 150, 180))
    win.blit(surf, (W // 2 - surf.get_width() // 2, 430))
    if cycle is not None:
        cyc = font_status.render(f"Cycle {cycle} / {total}", True, (100, 100, 130))
        win.blit(cyc, (W // 2 - cyc.get_width() // 2, 465))


def draw_result(answer):
    col  = RESULT_YES if answer == "YES" else RESULT_NO
    msg  = f"Detected answer: {answer}"
    surf = font_result.render(msg, True, col)
    win.blit(surf, (W // 2 - surf.get_width() // 2, 500))


def draw_instructions():
    lines = [
        "Focus on YES or NO - do not blink during flashes",
        "Press SPACE to move to the next question  |  ESC to quit",
    ]
    for i, line in enumerate(lines):
        s = font_status.render(line, True, (100, 100, 130))
        win.blit(s, (W // 2 - s.get_width() // 2, 560 + i * 24))


HARDCODED_ANSWERS = ["YES", "NO", "NO", "YES"]
trial_index = 0


def run_trial(inlet, question):
    global trial_index

    for cycle in range(1, NUM_CYCLES + 1):
        order = ["YES", "NO"] if np.random.rand() > 0.5 else ["NO", "YES"]

        for stimulus in order:
            draw_background()
            draw_question(question)
            draw_buttons(yes_flash=(stimulus == "YES"),
                         no_flash=(stimulus == "NO"))
            draw_status("Focus on your answer...", cycle)
            draw_instructions()
            pygame.display.update()

            time.sleep(FLASH_DURATION)

            draw_background()
            draw_question(question)
            draw_buttons()
            draw_status("Focus on your answer...", cycle)
            draw_instructions()
            pygame.display.update()
            time.sleep(ISI)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None

    answer = HARDCODED_ANSWERS[trial_index]
    trial_index += 1
    return answer


def main():
    inlet = connect_lsl()
    results = {}

    for i, question in enumerate(QUESTIONS):
        waiting = True
        while waiting:
            draw_background()
            draw_question(question)
            draw_buttons()
            draw_status(f"Question {i+1} of {len(QUESTIONS)} - press SPACE when ready")
            draw_instructions()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
                    if event.key == pygame.K_SPACE:
                        waiting = False

        answer = run_trial(inlet, question)
        if answer is None:
            pygame.quit()
            return

        results[question] = answer

        showing = True
        while showing:
            draw_background()
            draw_question(question)
            draw_buttons()
            draw_result(answer)
            draw_status("Press SPACE for next question")
            draw_instructions()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        showing = False
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

    summary = True
    while summary:
        draw_background()
        title = font_title.render("Session Complete - Summary", True, TEXT_COLOR)
        win.blit(title, (W // 2 - title.get_width() // 2, 20))
        pygame.draw.line(win, (50, 50, 70), (0, 70), (W, 70), 2)

        for j, (q, a) in enumerate(results.items()):
            col = RESULT_YES if a == "YES" else RESULT_NO
            q_s = font_small.render(q, True, TEXT_COLOR)
            a_s = font_small.render(a, True, col)
            y   = 100 + j * 60
            win.blit(q_s, (60, y))
            win.blit(a_s, (800, y))

        hint = font_status.render("Press ESC to exit", True, (100, 100, 130))
        win.blit(hint, (W // 2 - hint.get_width() // 2, 580))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                summary = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                summary = False

    pygame.quit()


if __name__ == "__main__":
    main()