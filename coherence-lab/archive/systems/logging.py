"""
Training Logger - Standardized Output Formatting

Consistent logging format across all phases.
Handles epoch output, topic displays, exam results, and JSON history.

Usage:
    logger = TrainingLogger(run_id='20241208_123456', log_dir='data/runs')
    logger.epoch_start(epoch=1, day=1)
    logger.log_train_metrics(loss=0.5, acc=0.85)
    logger.log_topic(name='alternating', acc=0.95, level=5, xp=250, ...)
    logger.log_exam_result(topic='alternating', passed=True, score=0.88, ...)
    logger.epoch_end()
    logger.save()
"""

import json
import sys
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    Standardized training output and history logging.

    Provides consistent formatting across phases and saves JSON logs.
    """

    def __init__(self, script_name, version, run_id=None, log_dir='data/runs'):
        self.script_name = script_name
        self.version = version
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = []
        self.current_epoch = {}
        self.metadata = {}

    def set_metadata(self, **kwargs):
        """Set run metadata (args, topics, etc.)."""
        self.metadata.update(kwargs)

    def epoch_start(self, epoch, day=None):
        """Start a new epoch."""
        self.current_epoch = {'epoch': epoch}
        day_str = f"Day {day} " if day else ""
        print(f"\n{day_str}(Epoch {epoch:2d})", flush=True)

    def log_train_metrics(self, loss, acc, **kwargs):
        """Log training metrics."""
        print(f"  Train: loss={loss:.4f}, acc={acc:.1%}")
        self.current_epoch['train_loss'] = loss
        self.current_epoch['train_acc'] = acc
        self.current_epoch.update(kwargs)

    def log_val_metrics(self, acc, **kwargs):
        """Log validation metrics."""
        print(f"  Val: acc={acc:.1%}")
        self.current_epoch['val_acc'] = acc
        self.current_epoch.update(kwargs)

    def log_show_stats(self, show_rate, reasons, approval_rate=None):
        """Log showing behavior stats."""
        print(f"  Shows: {show_rate:.1%} of answers shown to teacher")
        print(f"  Show reasons: {reasons}")
        self.current_epoch['show_rate'] = show_rate
        self.current_epoch['show_reasons'] = reasons
        if approval_rate is not None:
            self.current_epoch['approval_rate'] = approval_rate

    def log_development(self, internalization, trust, **kwargs):
        """Log developmental state."""
        print(f"  Internalization: {internalization:.1%}, Trust: {trust:.1%}")
        self.current_epoch['internalization'] = internalization
        self.current_epoch['trust'] = trust
        self.current_epoch.update(kwargs)

    def log_goals(self, teacher_goal, student_est, impressed, highest=None, cal_rate=None):
        """Log goal-setting metrics."""
        parts = [f"current={teacher_goal}"]
        if highest is not None:
            parts.append(f"best={highest}")
        parts.append(f"student_est={student_est:.1f}")
        parts.append(f"impressed={impressed:.0%}")
        print(f"  Goals: {', '.join(parts)}")

        self.current_epoch['teacher_goal'] = teacher_goal
        self.current_epoch['student_goal_estimate'] = student_est
        self.current_epoch['teacher_impressedness'] = impressed
        if highest is not None:
            self.current_epoch['highest_goal_achieved'] = highest
        if cal_rate is not None:
            self.current_epoch['goal_calibration_rate'] = cal_rate

    def log_xp_summary(self, total_xp, avg_level):
        """Log XP summary."""
        print(f"  XP: total={total_xp:.0f}, avg_level={avg_level:.1f}")
        self.current_epoch['total_xp'] = total_xp
        self.current_epoch['avg_level'] = avg_level

    def log_topic_header(self):
        """Print topic section header."""
        print("  Per-pattern:")

    def log_topic(self, name, acc, cal_status, level, progress, xp, graduated=False):
        """Log a single topic's stats with visual formatting."""
        # Accuracy symbol
        acc_symbol = "O" if acc >= 0.95 else ("o" if acc >= 0.85 else ".")

        # Calibration symbol
        cal_symbols = {'calibrated': 'C', 'guessing': '?', 'overconfident': '!', 'unknown': '.'}
        cal_symbol = cal_symbols.get(cal_status, '.')

        # Level bar: █ for each level, ░ for progress to next
        level_bar = "█" * level + ("░" if progress > 0.5 else "") + "·" * (10 - level - (1 if progress > 0.5 else 0))

        # Graduation marker
        grad_mark = " ✓" if graduated else ""

        print(f"    {name:15s}: {acc:.1%} {acc_symbol} {cal_symbol} L{level:2d} {level_bar} ({xp:.0f}xp){grad_mark}")

    def log_exam_header(self):
        """Print exam section header."""
        print("  Exams:")

    def log_exam_result(self, topic, passed, score, threshold, new_level=None, cooldown=None, graduated=False):
        """Log an exam result."""
        if passed:
            status = f"Ready for L{new_level}"
            if graduated:
                status += " - GRADUATED!"
            print(f"    {topic:15s}: {score:.0%} >= {threshold:.0%} - {status}")
        else:
            print(f"    {topic:15s}: {score:.0%} < {threshold:.0%} - More practice needed (cooldown: {cooldown} epochs)")

    def log_graduated_count(self, count, total):
        """Log graduated topic count."""
        if count > 0:
            print(f"  Graduated: {count}/{total} topics")

    def log_checkpoint_saved(self, n_topics=None):
        """Log checkpoint save."""
        if n_topics:
            print(f"  [Registry saved: {n_topics} topics]")
        else:
            print(f"  [Checkpoint saved]")

    def log_not_ready(self, issues):
        """Log why not ready for graduation."""
        print(f"  [Not ready: {', '.join(issues)}]")

    def add_to_history(self, **kwargs):
        """Add extra data to current epoch."""
        self.current_epoch.update(kwargs)

    def epoch_end(self):
        """End current epoch and add to history."""
        self.history.append(self.current_epoch.copy())
        sys.stdout.flush()

    def save(self, best_acc=None, final_trust=None, final_int=None, extra=None):
        """Save run log to JSON."""
        log_path = self.log_dir / f'{self.script_name}_{self.run_id}.json'

        run_log = {
            'script': f'{self.script_name}.py',
            'version': self.version,
            'run_id': self.run_id,
            **self.metadata,
            'epochs_completed': len(self.history),
            'history': self.history
        }

        if best_acc is not None:
            run_log['best_acc'] = best_acc
        if final_trust is not None:
            run_log['final_trust'] = final_trust
        if final_int is not None:
            run_log['final_internalization'] = final_int
        if extra:
            run_log.update(extra)

        with open(log_path, 'w') as f:
            json.dump(run_log, f, indent=2)

        print(f"Run log saved: {log_path}")
        return log_path

    def print_final_summary(self, best_acc, trust, int_level):
        """Print final run summary."""
        print("\n" + "=" * 70)
        print(f"Best accuracy: {best_acc:.1%}")
        print(f"Final trust: {trust:.1%}")
        print(f"Final internalization: {int_level:.1%}")

    def print_header(self, title, **kwargs):
        """Print run header."""
        print("=" * 70)
        print(title)
        for k, v in kwargs.items():
            print(f"{k}: {v}")
        print("=" * 70)


def format_level_bar(level, progress, max_level=10):
    """
    Format a visual level bar.

    Args:
        level: Current level (0-max_level)
        progress: Progress to next level (0-1)
        max_level: Maximum level

    Returns:
        String like "█████░····" for L5 with 60% progress
    """
    filled = "█" * level
    partial = "░" if progress > 0.5 else ""
    empty = "·" * (max_level - level - (1 if progress > 0.5 else 0))
    return filled + partial + empty
