"""
Relational Architecture: Self, Other, World + Temporality
Coherence Lab - Emergence Project

Establishing relational dynamics from the start:
- Self-model: "I am the learner, I have my own state"
- Other-model: "The teacher is separate, knows things I don't"
- World-model: "Reality exists independently of us both"
- Temporal-model: "I exist through time, experiences have continuity"

The first teachers are parents - trusted others who have your wellbeing in mind.
Trust enables learning in ways pure instruction doesn't.

Key principles:
- Separate representations for self and other
- Explicit communication channel (messages, not shared activations)
- Theory of mind: understanding the other has different knowledge
- Attachment/trust: the relationship that makes guidance effective
- Temporal continuity: experiences are timestamped and form threads
- Identity persistence: "I am the same entity that experienced that"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import argparse
from collections import deque
import time
import uuid


# =============================================================================
# TEMPORAL MODEL
# =============================================================================

class TemporalModel(nn.Module):
    """
    Temporal continuity and episodic memory.

    Key concepts:
    - Timestamp: when did this happen?
    - Episode ID: unique identifier for this experience
    - Day boundaries: rhythm of awake/sleep
    - Threads: experiences grouped by relationship, not just time

    Sleep = consolidation phase (training, weight updates)
    Awake = experience phase (inference, gathering experiences)
    """

    def __init__(self, d_model=64, max_episodes=1000, max_threads=32):
        super().__init__()
        self.d_model = d_model
        self.max_episodes = max_episodes

        # Episode embedding (each experience gets an ID)
        self.episode_embed = nn.Embedding(max_episodes, d_model)

        # Time encoding (continuous, like positional but for time)
        self.time_encoder = nn.Sequential(
            nn.Linear(4, d_model // 2),  # [day, hour, minute, second] normalized
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Thread representation (relational grouping of experiences)
        self.thread_embed = nn.Embedding(max_threads, d_model)
        self.thread_classifier = nn.Linear(d_model, max_threads)

        # Episodic memory buffer (experiences waiting for consolidation)
        self.register_buffer('episode_buffer', torch.zeros(max_episodes, d_model))
        self.register_buffer('episode_timestamps', torch.zeros(max_episodes, 4))
        self.register_buffer('episode_count', torch.tensor(0))
        self.register_buffer('current_day', torch.tensor(0))

        # Consolidated memory (after "sleep")
        self.consolidated_memory = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True
        )
        self.register_buffer('consolidated_state', torch.zeros(1, 1, d_model))

        # Day/night cycle
        self.is_awake = True

    def get_timestamp(self):
        """Get current time as normalized vector."""
        t = time.localtime()
        # Normalize to [0, 1] range
        return torch.tensor([
            t.tm_yday / 365,  # day of year
            t.tm_hour / 24,   # hour
            t.tm_min / 60,    # minute
            t.tm_sec / 60     # second
        ], dtype=torch.float)

    def create_episode(self, experience, timestamp=None):
        """
        Record an experience with timestamp and unique ID.

        Args:
            experience: [batch, d_model] the experience representation
            timestamp: optional explicit timestamp

        Returns:
            episode_id: unique identifier for this experience
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        batch_size = experience.size(0)
        episode_ids = []

        # Use no_grad to prevent inplace modification issues during backward
        with torch.no_grad():
            for i in range(batch_size):
                idx = self.episode_count.item() % self.max_episodes
                self.episode_buffer[idx].copy_(experience[i].detach())
                self.episode_timestamps[idx].copy_(timestamp)
                self.episode_count.add_(1)
                episode_ids.append(idx)

        return torch.tensor(episode_ids, device=experience.device)

    def encode_time(self, timestamp):
        """Encode timestamp into temporal representation."""
        if timestamp.dim() == 1:
            timestamp = timestamp.unsqueeze(0)
        return self.time_encoder(timestamp)

    def assign_thread(self, experience):
        """
        Assign experience to a relational thread.

        Like "that series of interactions with the teacher about patterns"
        or "that frustrating day when nothing worked"
        """
        thread_logits = self.thread_classifier(experience)
        thread_idx = thread_logits.argmax(dim=-1)
        return thread_idx, self.thread_embed(thread_idx)

    def recall_from_thread(self, thread_idx, n_recent=5):
        """
        Recall recent experiences from a thread.

        "What happened before in this context?"
        """
        # Get episodes assigned to this thread
        # (simplified: just return recent from buffer)
        count = min(n_recent, self.episode_count.item())
        if count == 0:
            return None
        start = max(0, self.episode_count.item() - count)
        return self.episode_buffer[start:start + count]

    def sleep(self):
        """
        Sleep phase: consolidate episodic memory into long-term.

        This is when learning actually happens - experiences are
        replayed and integrated.
        """
        self.is_awake = False

        # Get all buffered episodes
        count = min(self.episode_count.item(), self.max_episodes)
        if count == 0:
            return

        with torch.no_grad():
            episodes = self.episode_buffer[:count].unsqueeze(0)  # [1, count, d_model]

            # Process through consolidation
            _, new_state = self.consolidated_memory(episodes, self.consolidated_state)
            self.consolidated_state.copy_(new_state)

            # Advance to new day
            self.current_day.add_(1)

            # Clear episode buffer (new day, fresh experiences)
            self.episode_buffer.zero_()
            self.episode_count.zero_()

    def wake(self):
        """Wake up with consolidated knowledge from previous days."""
        self.is_awake = True
        return self.consolidated_state.squeeze(0)

    def get_temporal_context(self, experience):
        """
        Get full temporal context for an experience.

        Returns:
            - Time encoding
            - Thread membership
            - Relevant memories from same thread
            - Consolidated knowledge
        """
        timestamp = self.get_timestamp().to(experience.device)
        time_enc = self.encode_time(timestamp)

        thread_idx, thread_emb = self.assign_thread(experience)
        related_memories = self.recall_from_thread(thread_idx)

        return {
            'time_encoding': time_enc,
            'thread_embedding': thread_emb,
            'related_memories': related_memories,
            'consolidated_knowledge': self.consolidated_state.squeeze(0),
            'current_day': self.current_day.item(),
            'is_awake': self.is_awake
        }


# =============================================================================
# SELF MODEL
# =============================================================================

class SelfModel(nn.Module):
    """
    The learner's representation of itself.

    "I am confused" / "I am confident" / "I am stuck"
    Introspection on own state, separate from external perception.
    """

    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model

        # Internal state representation
        self.state_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Emotional/motivational state
        self.confidence = nn.Linear(d_model, 1)  # How sure am I?
        self.frustration = nn.Linear(d_model, 1)  # How stuck am I?
        self.curiosity = nn.Linear(d_model, 1)   # How interested am I?

        # Self-narrative: what am I trying to do?
        self.goal_representation = nn.Linear(d_model, d_model)

        # Memory of recent experiences
        self.experience_memory = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True
        )

    def forward(self, cognitive_state, goal=None):
        """
        Introspect on current state.

        Returns self-representation including:
        - Encoded internal state
        - Emotional readings (confidence, frustration, curiosity)
        - Goal representation
        """
        internal = self.state_encoder(cognitive_state)

        emotions = {
            'confidence': torch.sigmoid(self.confidence(internal)),
            'frustration': torch.sigmoid(self.frustration(internal)),
            'curiosity': torch.sigmoid(self.curiosity(internal))
        }

        if goal is not None:
            goal_rep = self.goal_representation(goal)
        else:
            goal_rep = torch.zeros_like(internal)

        return {
            'internal_state': internal,
            'emotions': emotions,
            'goal': goal_rep
        }

    def update_memory(self, experience, memory_state=None):
        """Add experience to autobiographical memory."""
        exp = experience.unsqueeze(1) if experience.dim() == 2 else experience
        if memory_state is None:
            memory_state = torch.zeros(1, experience.size(0), self.d_model,
                                       device=experience.device)
        _, new_memory = self.experience_memory(exp, memory_state)
        return new_memory


# =============================================================================
# OTHER MODEL (Theory of Mind)
# =============================================================================

class OtherModel(nn.Module):
    """
    The learner's representation of the teacher.

    "They know something I don't"
    "They want to help me"
    "I can trust their guidance"

    Theory of mind: understanding the other has different knowledge.

    Over time, the external teacher becomes internalized as conscience/inner guide.
    """

    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model

        # Representation of the other's knowledge state
        self.knowledge_model = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Trust/attachment level
        self.trust = nn.Parameter(torch.tensor(0.5))  # Learned trust level

        # What does the other want? (inferred intent)
        self.intent_model = nn.Linear(d_model, d_model)

        # History of interactions
        self.interaction_memory = nn.GRU(
            input_size=d_model * 2,  # message + outcome
            hidden_size=d_model,
            batch_first=True
        )

        # INTERNALIZATION: The teacher voice that lives inside
        # Starts empty, grows through experience
        self.internalized_teacher = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),  # bounded, like conscience
            nn.Linear(d_model, d_model)
        )

        # How internalized is the teacher? (development stage)
        self.internalization_level = nn.Parameter(torch.tensor(0.0))

        # Memory of teacher patterns (what would they say?)
        self.register_buffer('teacher_pattern_memory', torch.zeros(1, d_model))

    def forward(self, message_from_other, self_state):
        """
        Model the other based on their message and our state.

        Args:
            message_from_other: what the teacher communicated
            self_state: our current internal state

        Returns:
            Representation of the other including inferred knowledge and intent
        """
        # What do they know that I don't?
        knowledge_gap = self.knowledge_model(message_from_other - self_state)

        # What do they want? (inferred from message)
        inferred_intent = self.intent_model(message_from_other)

        # Current trust level
        trust_level = torch.sigmoid(self.trust)

        return {
            'knowledge_gap': knowledge_gap,
            'intent': inferred_intent,
            'trust': trust_level
        }

    def update_trust(self, outcome_was_good):
        """
        Update trust based on outcomes of following guidance.

        If following the teacher's guidance led to good outcomes,
        trust increases. If not, trust decreases (but doesn't go negative).
        """
        with torch.no_grad():
            delta = 0.1 if outcome_was_good else -0.05
            self.trust.data = torch.clamp(self.trust.data + delta, 0.0, 1.0)

    def internalize(self, teacher_message, outcome):
        """
        Internalize the teacher's pattern when guidance leads to good outcomes.

        The external teacher gradually becomes the inner voice.
        """
        with torch.no_grad():
            if outcome > 0.5:  # Good outcome
                # Blend teacher message into internal pattern memory
                alpha = 0.1
                self.teacher_pattern_memory = (1 - alpha) * self.teacher_pattern_memory + \
                                               alpha * teacher_message.mean(dim=0, keepdim=True)
                # Increase internalization level
                self.internalization_level.data = torch.clamp(
                    self.internalization_level.data + 0.01, 0.0, 1.0
                )

    def consult_inner_guide(self, self_state):
        """
        Consult the internalized teacher (conscience).

        "What would my teacher say about this?"

        Returns guidance based on accumulated patterns, weighted by
        how internalized the teacher has become.
        """
        # Generate what the internalized teacher would say
        inner_voice = self.internalized_teacher(self_state)

        # Weight by internalization level
        # Early: mostly zeros (no internalized guide yet)
        # Later: strong inner voice
        int_level = torch.sigmoid(self.internalization_level)

        # Also incorporate accumulated teacher patterns
        pattern_guidance = self.teacher_pattern_memory.expand(self_state.size(0), -1)

        guidance = int_level * (0.7 * inner_voice + 0.3 * pattern_guidance)

        return {
            'guidance': guidance,
            'internalization_level': int_level,
            'is_mature': int_level > 0.7  # Can rely on inner guide
        }


# =============================================================================
# COMMUNICATION CHANNEL
# =============================================================================

class CommunicationChannel(nn.Module):
    """
    Explicit message passing between learner and teacher.

    Not shared activations - actual messages that must be encoded/decoded.
    """

    def __init__(self, d_model=64, message_types=4):
        super().__init__()
        self.d_model = d_model

        # Message types
        self.message_types = ['question', 'answer', 'hint', 'encouragement']

        # Encode message for sending
        self.encoder = nn.Sequential(
            nn.Linear(d_model + message_types, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Decode received message
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model + message_types)
        )

        # Message type embeddings
        self.type_embed = nn.Embedding(message_types, message_types)

    def send(self, content, message_type_idx):
        """Encode a message for transmission."""
        batch_size = content.size(0)
        type_vec = self.type_embed(
            torch.full((batch_size,), message_type_idx, device=content.device)
        )
        combined = torch.cat([content, type_vec], dim=-1)
        return self.encoder(combined)

    def receive(self, message):
        """Decode a received message."""
        decoded = self.decoder(message)
        content = decoded[:, :self.d_model]
        type_logits = decoded[:, self.d_model:]
        return content, type_logits


# =============================================================================
# LEARNER (with Self-model)
# =============================================================================

class RelationalLearner(nn.Module):
    """
    A learner with explicit self-model, capacity for relationship, and temporal continuity.
    """

    def __init__(self, vocab_size=26, d_model=64, max_seq_len=12,
                 n_heads=4, n_think_steps=5):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_think_steps = n_think_steps

        # Perception (world interaction)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )

        # Self-model
        self.self_model = SelfModel(d_model)

        # Other-model (for teacher)
        self.other_model = OtherModel(d_model)

        # Temporal model (continuity through time)
        self.temporal_model = TemporalModel(d_model)

        # Communication
        self.comm_channel = CommunicationChannel(d_model)

        # Thinking - now includes temporal context
        self.step_embed = nn.Embedding(n_think_steps, d_model)
        self.think_layer = nn.GRU(
            input_size=d_model * 4,  # observation + self + other + temporal
            hidden_size=d_model,
            batch_first=True
        )

        # Action (attention-based look)
        self.look_query = nn.Linear(d_model * 2, d_model)
        self.look_key = nn.Linear(d_model, d_model)
        self.look_value = nn.Linear(d_model, d_model)

        # Output
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )

        # Pattern classifier (auxiliary)
        self.pattern_head = nn.Linear(d_model, 4)

    def perceive(self, tokens, seq_lens=None):
        """Perceive the world (encode sequence)."""
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        return self.encoder(x)

    def look(self, encoded, state, step_idx, seq_lens=None):
        """Attend to a position in the perceived world."""
        batch_size = encoded.size(0)
        device = encoded.device

        step_tensor = torch.full((batch_size,), step_idx, dtype=torch.long, device=device)
        step_emb = self.step_embed(step_tensor)

        Q = self.look_query(torch.cat([state, step_emb], dim=-1)).unsqueeze(1)
        K = self.look_key(encoded)
        V = self.look_value(encoded)

        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        if seq_lens is not None:
            for i, slen in enumerate(seq_lens):
                scores[i, 0, slen:] = float('-inf')

        attention = F.softmax(scores, dim=-1)
        observation = torch.bmm(attention, V).squeeze(1)
        return observation, attention.squeeze(1)

    def ask_for_help(self, self_state):
        """
        Formulate a question to the teacher.

        When stuck, the learner can explicitly ask for guidance.
        """
        return self.comm_channel.send(self_state['internal_state'], 0)  # type 0 = question

    def receive_guidance(self, message, self_state):
        """
        Receive and interpret guidance from teacher.

        Uses other-model to understand the message.
        """
        content, type_logits = self.comm_channel.receive(message)
        other_rep = self.other_model(content, self_state['internal_state'])
        return content, other_rep

    def forward(self, tokens, seq_lens=None, teacher_message=None, return_details=False):
        """
        Forward pass with self-awareness and optional teacher interaction.
        """
        batch_size = tokens.size(0)
        device = tokens.device

        # Perceive the world
        encoded = self.perceive(tokens, seq_lens)

        # Initialize cognitive state
        cognitive_state = torch.zeros(batch_size, self.d_model, device=device)

        # Track for self-model
        states_history = []

        # If we have a message from teacher, process it
        if teacher_message is not None:
            self_state = self.self_model(cognitive_state)
            guidance, other_rep = self.receive_guidance(teacher_message, self_state)
            # Weight guidance by trust
            trust = other_rep['trust']
            cognitive_state = cognitive_state + trust * guidance
        else:
            other_rep = None

        # Get temporal context (continuity from past experiences)
        temporal_ctx = self.temporal_model.get_temporal_context(cognitive_state)

        # Think steps
        for step in range(self.n_think_steps):
            # Self-reflection
            self_state = self.self_model(cognitive_state)

            # Look at the world
            observation, attention = self.look(encoded, cognitive_state, step, seq_lens)

            # Combine: observation + self-awareness + other + temporal
            if other_rep is not None:
                other_input = other_rep['knowledge_gap']
            else:
                other_input = torch.zeros_like(cognitive_state)

            # Temporal input: time encoding + consolidated knowledge
            temporal_input = temporal_ctx['time_encoding'].expand(batch_size, -1) + \
                           temporal_ctx['consolidated_knowledge'].expand(batch_size, -1)

            think_input = torch.cat([
                observation,
                self_state['internal_state'],
                other_input,
                temporal_input
            ], dim=-1).unsqueeze(1)

            # Update cognitive state
            _, hidden = self.think_layer(think_input, cognitive_state.unsqueeze(0))
            cognitive_state = hidden.squeeze(0)

            states_history.append(cognitive_state.clone())

        # Record this experience for later consolidation
        self.temporal_model.create_episode(cognitive_state)

        # Final self-state
        final_self = self.self_model(cognitive_state)

        # Output
        logits = self.output_head(cognitive_state)
        pattern_logits = self.pattern_head(cognitive_state)

        if return_details:
            return {
                'logits': logits,
                'pattern_logits': pattern_logits,
                'self_state': final_self,
                'other_rep': other_rep,
                'states_history': states_history
            }

        return logits, final_self['emotions']['confidence'], pattern_logits


# =============================================================================
# PROCESS EVALUATOR (Creative + Bad Habit Detection)
# =============================================================================

class ProcessEvaluator(nn.Module):
    """
    Evaluates HOW the learner arrives at answers, not just WHAT they answer.

    This is the meta-level teaching that shapes thinking patterns:
    - Reward creative/novel approaches that lead to correct answers
    - Catch bad habits: lucky guesses, rote patterns, overconfidence

    "Show your work for partial credit" - the process matters.
    """

    def __init__(self, d_model=64, history_len=50):
        super().__init__()
        self.d_model = d_model
        self.history_len = history_len

        # Encode the reasoning path (sequence of states during think steps)
        self.path_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True
        )

        # Detect novel approaches (path differs from typical successful paths)
        self.novelty_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # path + prototype
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

        # Store prototype successful paths per pattern type
        self.register_buffer('pattern_prototypes', torch.zeros(8, d_model))  # 8 pattern types max
        self.register_buffer('prototype_counts', torch.zeros(8))

        # Detect bad habits
        self.habit_classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # path + confidence + outcome
            nn.ReLU(),
            nn.Linear(d_model, 4)  # [good_reasoning, lucky_guess, rote_pattern, overconfident]
        )

        # Track learner behavior history for habit detection
        self.register_buffer('behavior_history', torch.zeros(history_len, 4))  # recent habit scores
        self.register_buffer('history_ptr', torch.tensor(0))

        # Efficiency metric: did they take a good path?
        self.efficiency_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def encode_reasoning_path(self, states_history):
        """
        Encode the sequence of cognitive states during think steps.

        Args:
            states_history: list of [batch, d_model] tensors from each think step
        """
        if len(states_history) == 0:
            return None

        # Stack into [batch, n_steps, d_model]
        path = torch.stack(states_history, dim=1)

        # Encode the full path
        _, path_encoding = self.path_encoder(path)
        return path_encoding.squeeze(0)  # [batch, d_model]

    def detect_creativity(self, path_encoding, pattern_type_idx):
        """
        Detect if the learner used a novel approach.

        Creative = correct answer via non-typical path.
        Only meaningful after we have prototype data.
        """
        batch_size = path_encoding.size(0)

        # Need enough examples to have a meaningful prototype
        if self.prototype_counts[pattern_type_idx].item() < 10:
            # Not enough data yet - don't claim creativity
            return torch.zeros(batch_size, 1, device=path_encoding.device)

        # Get prototype for this pattern type
        prototype = self.pattern_prototypes[pattern_type_idx].unsqueeze(0).expand(batch_size, -1)

        # Compare path to prototype
        combined = torch.cat([path_encoding, prototype], dim=-1)
        novelty = self.novelty_detector(combined)

        return novelty

    def update_prototype(self, path_encoding, pattern_type_idx, was_correct):
        """
        Update prototype path for successful approaches.

        Only integrate correct approaches into the prototype.
        """
        with torch.no_grad():
            if was_correct.any():
                # Get correct examples
                correct_paths = path_encoding[was_correct].mean(dim=0)

                # Blend into prototype (exponential moving average)
                alpha = 0.1
                count = self.prototype_counts[pattern_type_idx].item()
                if count < 10:
                    alpha = 0.3  # Learn faster initially

                self.pattern_prototypes[pattern_type_idx] = (
                    (1 - alpha) * self.pattern_prototypes[pattern_type_idx] +
                    alpha * correct_paths
                )
                self.prototype_counts[pattern_type_idx] += was_correct.sum()

    def detect_bad_habits(self, path_encoding, confidence, was_correct):
        """
        Detect problematic learning patterns.

        Bad habits:
        - Lucky guess: low confidence but correct (didn't really understand)
        - Rote pattern: same path regardless of input (not actually reasoning)
        - Overconfident: high confidence but wrong (dangerous miscalibration)
        - Good reasoning: appropriate confidence matching outcome
        """
        batch_size = path_encoding.size(0)

        # Encode confidence as vector
        conf_expanded = confidence.expand(-1, self.d_model)

        # Encode outcome as vector
        outcome_expanded = was_correct.unsqueeze(-1).float().expand(-1, self.d_model)

        # Classify behavior
        combined = torch.cat([path_encoding, conf_expanded, outcome_expanded], dim=-1)
        habit_logits = self.habit_classifier(combined)
        habit_probs = F.softmax(habit_logits, dim=-1)

        return {
            'good_reasoning': habit_probs[:, 0],
            'lucky_guess': habit_probs[:, 1],
            'rote_pattern': habit_probs[:, 2],
            'overconfident': habit_probs[:, 3],
            'raw_logits': habit_logits
        }

    def update_history(self, habit_scores):
        """Track behavior patterns over time."""
        with torch.no_grad():
            # Average scores from this batch
            avg_scores = torch.tensor([
                habit_scores['good_reasoning'].mean().item(),
                habit_scores['lucky_guess'].mean().item(),
                habit_scores['rote_pattern'].mean().item(),
                habit_scores['overconfident'].mean().item()
            ], device=self.behavior_history.device)

            # Add to circular buffer
            ptr = self.history_ptr.item() % self.history_len
            self.behavior_history[ptr] = avg_scores
            self.history_ptr += 1

    def get_habit_trends(self):
        """
        Analyze behavior trends over recent history.

        Returns persistent patterns that need addressing.
        """
        count = min(self.history_ptr.item(), self.history_len)
        if count < 5:
            return None  # Not enough history

        recent = self.behavior_history[:count]
        avg = recent.mean(dim=0)

        return {
            'good_reasoning_trend': avg[0].item(),
            'lucky_guess_trend': avg[1].item(),
            'rote_pattern_trend': avg[2].item(),
            'overconfident_trend': avg[3].item(),
            'primary_concern': ['good_reasoning', 'lucky_guess', 'rote_pattern', 'overconfident'][avg.argmax().item()]
        }

    def evaluate_efficiency(self, path_encoding):
        """Score how efficient the reasoning path was."""
        return self.efficiency_scorer(path_encoding)

    def full_evaluation(self, states_history, confidence, was_correct, pattern_type_idx):
        """
        Complete process evaluation.

        Returns:
            - creativity_score: how novel was the approach
            - habit_scores: breakdown of reasoning quality
            - efficiency_score: how direct was the path
            - recommendations: what should teacher reward/discourage
        """
        path_encoding = self.encode_reasoning_path(states_history)
        if path_encoding is None:
            return None

        creativity = self.detect_creativity(path_encoding, pattern_type_idx)
        habits = self.detect_bad_habits(path_encoding, confidence, was_correct)
        efficiency = self.evaluate_efficiency(path_encoding)

        # Update prototype with successful paths
        self.update_prototype(path_encoding, pattern_type_idx, was_correct)

        # Update behavior history
        self.update_history(habits)

        # Generate recommendations
        recommendations = {
            'reward_creativity': (creativity > 0.7) & was_correct,
            'warn_lucky_guess': habits['lucky_guess'] > 0.5,
            'warn_rote': habits['rote_pattern'] > 0.5,
            'warn_overconfident': habits['overconfident'] > 0.5,
            'praise_good_reasoning': habits['good_reasoning'] > 0.7
        }

        return {
            'creativity': creativity,
            'habits': habits,
            'efficiency': efficiency,
            'recommendations': recommendations,
            'path_encoding': path_encoding
        }


# =============================================================================
# TEACHER (as Separate Entity)
# =============================================================================

class RelationalTeacher(nn.Module):
    """
    The teacher as a genuinely separate entity.

    Has its own state, knowledge, and goals.
    Communicates through explicit messages, not shared activations.
    Acts like a parent: trusted, patient, focused on learner's growth.

    Now also watches for:
    - Creative approaches to reward
    - Bad habits to gently correct
    """

    def __init__(self, d_model=64, patience=5):
        super().__init__()
        self.d_model = d_model
        self.patience = patience

        # Teacher's knowledge (ground truth understanding)
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Communication channel (shared with learner)
        self.comm_channel = CommunicationChannel(d_model)

        # Generate different message types
        self.hint_generator = nn.Linear(d_model * 2, d_model)  # state + knowledge
        self.encouragement_generator = nn.Linear(d_model, d_model)
        self.explanation_generator = nn.Linear(d_model * 2, d_model)

        # Observe learner state (but can't directly access it)
        self.learner_observer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Track interaction history
        self.interaction_count = 0

        # Process evaluator for watching HOW learner thinks
        self.process_evaluator = ProcessEvaluator(d_model)

        # Generate process-specific feedback
        self.creativity_reward_generator = nn.Linear(d_model, d_model)
        self.habit_correction_generator = nn.Linear(d_model * 2, d_model)  # habit + state

    def observe_learner(self, learner_visible_state):
        """
        Observe what the learner is showing (not their internal state).

        The teacher can see behavior, not read minds.
        """
        return self.learner_observer(learner_visible_state)

    def generate_hint(self, learner_observed, correct_direction):
        """
        Generate a hint toward the correct answer.

        Not the answer itself, but a nudge in the right direction.
        """
        combined = torch.cat([learner_observed, correct_direction], dim=-1)
        hint_content = self.hint_generator(combined)
        return self.comm_channel.send(hint_content, 2)  # type 2 = hint

    def generate_encouragement(self, learner_observed):
        """
        Generate encouragement.

        "You're on the right track" / "Keep trying"
        """
        enc_content = self.encouragement_generator(learner_observed)
        return self.comm_channel.send(enc_content, 3)  # type 3 = encouragement

    def provide_explanation(self, learner_observed, correct_answer_rep):
        """
        Explain why something is correct (after the fact).

        Not punishment for wrong, but illumination of right.
        """
        combined = torch.cat([learner_observed, correct_answer_rep], dim=-1)
        exp_content = self.explanation_generator(combined)
        return self.comm_channel.send(exp_content, 1)  # type 1 = answer/explanation

    def generate_creativity_reward(self, learner_state):
        """
        Generate positive reinforcement for creative approaches.

        "That's a clever way to think about it!"
        """
        reward_content = self.creativity_reward_generator(learner_state)
        return self.comm_channel.send(reward_content, 3)  # encouragement type

    def generate_habit_correction(self, habit_type, learner_state):
        """
        Generate gentle correction for bad habits.

        Not punishment - redirection. "Let's slow down and think this through."
        """
        # Encode the habit type as a vector
        habit_encoding = torch.zeros_like(learner_state)
        if habit_type == 'lucky_guess':
            habit_encoding = habit_encoding + 0.3  # mild concern
        elif habit_type == 'rote_pattern':
            habit_encoding = habit_encoding - 0.3  # different concern
        elif habit_type == 'overconfident':
            habit_encoding = habit_encoding + 0.6  # more serious

        combined = torch.cat([habit_encoding, learner_state], dim=-1)
        correction_content = self.habit_correction_generator(combined)
        return self.comm_channel.send(correction_content, 2)  # hint type

    def evaluate_learner_process(self, states_history, confidence, was_correct, pattern_type_idx):
        """
        Full evaluation of HOW the learner arrived at their answer.

        This is the meta-level teaching that shapes thinking patterns.
        """
        return self.process_evaluator.full_evaluation(
            states_history, confidence, was_correct, pattern_type_idx
        )

    def decide_intervention(self, learner_emotions, process_eval=None):
        """
        Decide whether and how to intervene.

        Like a parent: intervene when stuck, encourage when frustrated,
        step back when confident and exploring.

        Now also considers HOW the learner is thinking:
        - Reward creativity
        - Gently correct bad habits
        """
        confidence = learner_emotions['confidence']
        frustration = learner_emotions['frustration']
        curiosity = learner_emotions['curiosity']

        # High frustration + low confidence = needs help
        needs_help = (frustration > 0.7) & (confidence < 0.3)

        # Low frustration + high curiosity = exploring, don't interrupt
        is_exploring = (frustration < 0.3) & (curiosity > 0.5)

        # Moderate everything = might benefit from encouragement
        needs_encouragement = (confidence > 0.3) & (confidence < 0.7)

        intervention = {
            'should_help': needs_help,
            'should_encourage': needs_encouragement & ~is_exploring,
            'should_step_back': is_exploring,
            # Process-related interventions
            'should_reward_creativity': torch.zeros_like(confidence).bool(),
            'should_correct_habit': torch.zeros_like(confidence).bool(),
            'habit_to_correct': None
        }

        # Add process-based interventions if we have evaluation
        if process_eval is not None:
            recs = process_eval['recommendations']

            # Reward creativity (positive reinforcement)
            intervention['should_reward_creativity'] = recs['reward_creativity'].squeeze(-1)

            # Check for bad habits (gentle correction)
            # Prioritize: overconfident > lucky_guess > rote_pattern
            if recs['warn_overconfident'].any():
                intervention['should_correct_habit'] = recs['warn_overconfident']
                intervention['habit_to_correct'] = 'overconfident'
            elif recs['warn_lucky_guess'].any():
                intervention['should_correct_habit'] = recs['warn_lucky_guess']
                intervention['habit_to_correct'] = 'lucky_guess'
            elif recs['warn_rote'].any():
                intervention['should_correct_habit'] = recs['warn_rote']
                intervention['habit_to_correct'] = 'rote_pattern'

            # Good reasoning gets praised too
            intervention['should_encourage'] = intervention['should_encourage'] | \
                                               recs['praise_good_reasoning']

        return intervention

    def get_habit_trends(self):
        """Get long-term habit trends for progress reporting."""
        return self.process_evaluator.get_habit_trends()


# =============================================================================
# INTEGRATED RELATIONAL SYSTEM
# =============================================================================

class RelationalSystem(nn.Module):
    """
    Complete system with proper self/other/world distinction.
    """

    def __init__(self, vocab_size=26, d_model=64, max_seq_len=12,
                 n_heads=4, n_think_steps=5):
        super().__init__()

        self.learner = RelationalLearner(
            vocab_size, d_model, max_seq_len, n_heads, n_think_steps
        )
        self.teacher = RelationalTeacher(d_model)

        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, tokens, seq_lens=None, pattern_types=None, targets=None,
                allow_teacher_interaction=True, return_details=False):
        """
        Forward pass with relational dynamics.

        Now includes process evaluation for creativity/habit detection.
        """
        device = tokens.device

        # First pass: learner tries on their own
        learner_output = self.learner(tokens, seq_lens, return_details=True)

        # Teacher observes learner's visible state
        learner_visible = learner_output['self_state']['internal_state']
        teacher_observation = self.teacher.observe_learner(learner_visible)

        # Process evaluation (need targets to know correctness)
        process_eval = None
        if targets is not None and 'states_history' in learner_output:
            preds = learner_output['logits'].argmax(dim=-1)
            was_correct = (preds == targets)
            confidence = learner_output['self_state']['emotions']['confidence']

            # Get pattern type index (use 0 as default for mixed batches)
            pattern_idx = 0  # Could be expanded for per-example

            process_eval = self.teacher.evaluate_learner_process(
                learner_output['states_history'],
                confidence,
                was_correct,
                pattern_idx
            )

        # Teacher decides on intervention (now with process info)
        intervention = self.teacher.decide_intervention(
            learner_output['self_state']['emotions'],
            process_eval
        )

        # Generate teacher message based on intervention type
        teacher_message = None
        if allow_teacher_interaction:
            if intervention['should_reward_creativity'].any():
                # Reward creative approaches
                teacher_message = self.teacher.generate_creativity_reward(teacher_observation)

            elif intervention['should_correct_habit'].any() and intervention['habit_to_correct']:
                # Gently correct bad habits
                teacher_message = self.teacher.generate_habit_correction(
                    intervention['habit_to_correct'],
                    teacher_observation
                )

            elif intervention['should_help'].any():
                # Standard help for stuck learner
                teacher_message = self.teacher.generate_encouragement(teacher_observation)

            # If teacher provided message, learner processes again with guidance
            if teacher_message is not None:
                learner_output = self.learner(tokens, seq_lens, teacher_message, return_details=True)

        if return_details:
            return {
                'logits': learner_output['logits'],
                'pattern_logits': learner_output['pattern_logits'],
                'learner_self': learner_output['self_state'],
                'intervention': intervention,
                'teacher_message': teacher_message,
                'process_eval': process_eval,
                'states_history': learner_output.get('states_history', [])
            }

        return learner_output['logits'], learner_output['self_state']['emotions']['confidence'], learner_output['pattern_logits']


# =============================================================================
# DATASET (same as before)
# =============================================================================

class PatternDataset(Dataset):
    def __init__(self, n_examples=50000, vocab_size=26, seed=None):
        if seed:
            random.seed(seed)
        self.vocab_size = vocab_size
        self.examples = []
        for _ in range(n_examples):
            pattern_type = random.choice(['alternating', 'repeating', 'incrementing', 'fixed_offset'])
            self.examples.append(self._generate(pattern_type))

    def _generate(self, pt):
        if pt == 'alternating':
            a, b = random.sample(range(self.vocab_size), 2)
            length = random.randint(4, 8)
            seq = [a if i % 2 == 0 else b for i in range(length)]
            target = a if length % 2 == 0 else b
        elif pt == 'repeating':
            a = random.randint(0, self.vocab_size - 1)
            length = random.randint(3, 7)
            seq = [a] * length
            target = a
        elif pt == 'incrementing':
            length = random.randint(3, 6)
            start = random.randint(0, max(0, self.vocab_size - length - 1))
            seq = [start + i for i in range(length)]
            target = start + length
        else:  # fixed_offset
            length = random.randint(3, 5)
            k = random.randint(1, 3)
            start = random.randint(0, max(0, self.vocab_size - k * length - 1))
            seq = [start + i * k for i in range(length)]
            target = start + length * k
        return {'sequence': seq, 'target': target, 'pattern_type': pt}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        padded = ex['sequence'] + [0] * (12 - len(ex['sequence']))
        return {
            'sequence': torch.tensor(padded[:12], dtype=torch.long),
            'target': torch.tensor(ex['target'], dtype=torch.long),
            'seq_len': len(ex['sequence']),
            'pattern_type': ex['pattern_type']
        }


def collate_fn(batch):
    return {
        'tokens': torch.stack([b['sequence'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'seq_len': [b['seq_len'] for b in batch],
        'pattern_type': [b['pattern_type'] for b in batch]
    }


# =============================================================================
# TRAINING WITH DAY/SLEEP CYCLES
# =============================================================================

def train_day(model, loader, optimizer, criterion, device, pattern_to_idx, val_loader=None):
    """
    One "day" of learning: experiencing, trying, receiving guidance.

    At the end of the day, we'll sleep to consolidate.

    Now includes process-based rewards:
    - Bonus for creativity
    - Penalty for bad habits (but gentle - redirection not punishment)
    - Validation spot-checks to detect generalization gaps (overconfidence)
    """
    model.train()

    # Wake up with consolidated knowledge from previous days
    model.learner.temporal_model.wake()

    total_loss, total_correct, total_samples = 0, 0, 0
    intervention_count = 0
    creativity_count = 0
    habit_corrections = {'lucky_guess': 0, 'rote_pattern': 0, 'overconfident': 0}

    # Track running train accuracy for generalization gap detection
    running_train_acc = 0.0
    generalization_gap = 0.0
    val_iter = iter(val_loader) if val_loader else None

    for batch_idx, batch in enumerate(loader):
        tokens = batch['tokens'].to(device)
        targets = batch['target'].to(device)
        seq_lens = batch['seq_len']
        pattern_types = batch['pattern_type']

        optimizer.zero_grad()

        # Pass targets for process evaluation
        details = model(tokens, seq_lens, targets=targets, return_details=True)

        # Main loss
        main_loss = criterion(details['logits'], targets)

        # Pattern classification auxiliary
        pattern_targets = torch.tensor([pattern_to_idx[p] for p in pattern_types], device=device)
        aux_loss = criterion(details['pattern_logits'], pattern_targets)

        # Self-awareness loss: confidence should match correctness
        preds = details['logits'].argmax(dim=-1)
        correct = (preds == targets).float()
        conf = details['learner_self']['emotions']['confidence'].squeeze()
        conf_loss = F.binary_cross_entropy(conf, correct)

        # === VALIDATION SPOT-CHECK: Teacher gives pop quizzes ===
        if val_iter and batch_idx % 50 == 0 and batch_idx > 0:
            model.eval()
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)

            with torch.no_grad():
                val_tokens = val_batch['tokens'].to(device)
                val_targets = val_batch['target'].to(device)
                val_logits, val_conf, _ = model(val_tokens, val_batch['seq_len'],
                                                 allow_teacher_interaction=False)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == val_targets).float().mean().item()

                # Detect generalization gap
                running_train_acc = total_correct / max(1, total_samples)
                generalization_gap = running_train_acc - val_acc

                # If big gap AND high confidence on wrong answers = overconfidence
                if generalization_gap > 0.15:
                    val_wrong = (val_preds != val_targets)
                    if val_wrong.any():
                        overconf_on_val = val_conf[val_wrong].mean().item()
                        if overconf_on_val > 0.6:
                            # Inject overconfidence penalty into next batch
                            # This makes the teacher aware of the generalization problem
                            habit_corrections['overconfident'] += val_wrong.sum().item()

            model.train()

        # Process-based rewards/penalties
        process_loss = torch.tensor(0.0, device=device)
        if details['process_eval'] is not None:
            habits = details['process_eval']['habits']

            # Encourage good reasoning (negative loss = reward)
            good_reasoning_bonus = -0.1 * habits['good_reasoning'].mean()

            # Discourage bad habits (positive loss = penalty, but gentle)
            lucky_guess_penalty = 0.05 * habits['lucky_guess'].mean()
            rote_penalty = 0.05 * habits['rote_pattern'].mean()
            overconfident_penalty = 0.1 * habits['overconfident'].mean()

            # EXTRA penalty if we detected generalization gap
            if generalization_gap > 0.15:
                overconfident_penalty += 0.2 * generalization_gap

            # Creativity bonus (only for correct answers)
            creativity = details['process_eval']['creativity']
            creativity_bonus = -0.15 * (creativity * correct.unsqueeze(-1)).mean()

            process_loss = (good_reasoning_bonus + lucky_guess_penalty +
                          rote_penalty + overconfident_penalty + creativity_bonus)

        # Combined loss
        loss = main_loss + 0.2 * aux_loss + 0.1 * conf_loss + process_loss
        loss.backward()
        optimizer.step()

        # Update internalization based on process quality, not just correctness
        if details['process_eval'] is not None:
            # Good process should strengthen internalization
            good_process = details['process_eval']['habits']['good_reasoning'].mean().item() > 0.5
            # DON'T internalize if we have a generalization gap!
            no_overfit = generalization_gap < 0.1
            if good_process and correct.mean().item() > 0.7 and no_overfit:
                # Teacher guidance led to good process AND good outcomes AND generalizes
                if details['teacher_message'] is not None:
                    model.learner.other_model.internalize(
                        details['teacher_message'],
                        torch.tensor(correct.mean().item())
                    )

        total_loss += main_loss.item() * tokens.size(0)
        total_correct += correct.sum().item()
        total_samples += tokens.size(0)

        # Track interventions
        if details['teacher_message'] is not None:
            intervention_count += details['intervention']['should_help'].sum().item()

        # Track process metrics
        if details['process_eval'] is not None:
            # Count actual creative instances (not just any() which triggers too often)
            creativity_count += details['intervention']['should_reward_creativity'].float().sum().item()

            if details['intervention']['habit_to_correct']:
                habit = details['intervention']['habit_to_correct']
                habit_corrections[habit] += details['intervention']['should_correct_habit'].sum().item()

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'interventions': intervention_count / total_samples,
        'creativity_rewards': creativity_count / total_samples,
        'habit_corrections': {k: v / total_samples for k, v in habit_corrections.items()},
        'generalization_gap': generalization_gap
    }


def evaluate(model, loader, device, pattern_to_idx):
    model.eval()
    total_correct, total_samples = 0, 0
    pattern_correct = {p: 0 for p in pattern_to_idx}
    pattern_total = {p: 0 for p in pattern_to_idx}

    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            targets = batch['target'].to(device)
            pattern_types = batch['pattern_type']

            logits, _, _ = model(tokens, batch['seq_len'], allow_teacher_interaction=False)
            preds = logits.argmax(dim=-1)
            correct = (preds == targets)

            total_correct += correct.sum().item()
            total_samples += len(preds)

            for i, pt in enumerate(pattern_types):
                pattern_total[pt] += 1
                if correct[i]:
                    pattern_correct[pt] += 1

    return {
        'accuracy': total_correct / total_samples,
        'per_pattern': {p: pattern_correct[p] / max(1, pattern_total[p]) for p in pattern_to_idx}
    }


def main(args):
    print("=" * 70)
    print("Relational Architecture: Self, Other, World")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    pattern_to_idx = {p: i for i, p in enumerate(['alternating', 'repeating', 'incrementing', 'fixed_offset'])}

    train_data = PatternDataset(n_examples=args.n_train, seed=42)
    val_data = PatternDataset(n_examples=args.n_val, seed=123)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    model = RelationalSystem(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_think_steps=args.n_think_steps
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"\nLearner has: SelfModel, OtherModel, CommunicationChannel")
    print(f"Teacher has: separate knowledge, patience, intervention logic")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining with relational dynamics...")
    print("-" * 70)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_day(model, train_loader, optimizer, criterion, device, pattern_to_idx, val_loader)

        # Sleep at end of day - consolidate experiences
        model.learner.temporal_model.sleep()
        val_metrics = evaluate(model, val_loader, device, pattern_to_idx)

        # Get developmental state
        day = model.learner.temporal_model.current_day.item()
        int_level = torch.sigmoid(model.learner.other_model.internalization_level).item()

        print(f"\nDay {day} (Epoch {epoch:2d})")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.1%}")
        print(f"  Val: acc={val_metrics['accuracy']:.1%}")
        gap = train_metrics.get('generalization_gap', 0)
        if gap > 0.1:
            print(f"  ⚠ Generalization gap: {gap:.1%} (train >> val)")
        print(f"  Teacher interventions: {train_metrics['interventions']:.1%}")
        print(f"  Internalization: {int_level:.1%}")

        # Process metrics
        print(f"  Process feedback:")
        print(f"    Creativity rewards: {train_metrics['creativity_rewards']:.1%}")
        print(f"    Habit corrections: {train_metrics['habit_corrections']}")

        # Habit trends (if enough history)
        trends = model.teacher.get_habit_trends()
        if trends:
            print(f"  Habit trends (rolling):")
            print(f"    Good reasoning: {trends['good_reasoning_trend']:.1%}")
            print(f"    Lucky guessing: {trends['lucky_guess_trend']:.1%}")
            print(f"    Primary concern: {trends['primary_concern']}")

        print("  Per-pattern:")
        for pt in pattern_to_idx:
            print(f"    {pt:15s}: {val_metrics['per_pattern'][pt]:.1%}")

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': best_acc},
                       Path(args.data_dir) / 'relational_model.pt')

        if val_metrics['accuracy'] >= 0.95:
            print(f"\n*** Relational system mastered patterns! ({val_metrics['accuracy']:.1%}) ***")
            break

    print(f"\nBest accuracy: {best_acc:.1%}")
    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--n-train', type=int, default=80000)
    parser.add_argument('--n-val', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-think-steps', type=int, default=5)
    args = parser.parse_args()
    main(args)
