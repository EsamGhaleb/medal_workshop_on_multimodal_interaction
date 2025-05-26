class Tier:
    ''' This class holds information for each tier'''
    def __init__(self, tier_name, from_ts, to_ts, value, duration):
        self.tier_name = tier_name
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.value = value
        self.duration = duration
    def __str__(self) -> str:
        return 'Tier name is {} with value \"{}\"'.format(self.tier_name, self.value)

class Turn:
    ''' This class holds information for each turn''' 
    def __init__(self, pair, speaker, turn_ID, target_turn, utterance, duration, 
    from_ts, to_ts, round, trial, target, director, correct_answer, given_answer, accuracy, dataset):
        self.speaker = speaker
        self.ID = turn_ID
        self.utterance = utterance
        self.lemmas_with_pos = ''
        self.pos_sequence = ''
        self.lemmas_sequence = ''
        self.text_lemma_pos = ''
        self.gesture = []
        self.duration = duration
        self.from_ts = from_ts
        self.to_ts = to_ts 
        self.target = target
        self.trial = trial
        self.round = round
        self.director = director
        self.correct_answer = correct_answer
        self.given_answer = given_answer
        self.accuracy = accuracy
        self.dataset = dataset
        self.target_turn = target_turn
        self.utterance_speech = []
        self.processed_utterance = ''
        self.pair = pair
    def __str__(self) -> str:
        return 'Speaker is {} with utterance \"{}\". The trial is {} where the director is {} talking about {}'.format(self.speaker, 
        self.utterance, self.trial, self.director, self.target)
    def set_lemmas_with_pos(self, lemmas_with_pos):
        self.lemmas_with_pos = lemmas_with_pos
    def set_pos_sequence(self, pos_sequence):
        self.pos_sequence = pos_sequence
    def set_lemmas_sequence(self, lemmas_sequence):
        self.lemmas_sequence = lemmas_sequence
    def set_text_lemma_pos(self, text_lemma_pos):
        self.text_lemma_pos = text_lemma_pos
    def set_ID(self, ID):
        self.ID = ID
    def set_target_turn(self, target_turn):
        self.target_turn = target_turn
    def set_utterance_speech(self, utterance_speech):
        self.utterance_speech = utterance_speech
    def set_processed_utterance(self, processed_utterance):
        self.processed_utterance = processed_utterance
    def set_gesture(self, gesture):
        self.gesture.append(gesture)
class Utterance:
    ''' This class holds information for each utterance'''
    def __init__(self, word, from_ts, to_ts, lemma='', pos='', lemma_pos=''):
        self.word = word
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.lemma = lemma
        self.pos = pos
        self.lemma_pos = lemma_pos
    def set_lemma(self, lemma):
        self.lemma = lemma
    def set_pos(self, pos):
        self.pos = pos
    def set_lemma_pos(self, lemma_pos):
        self.lemma_pos = lemma_pos
    def __str__(self) -> str:
        return 'word is \"{}\", from_ts is {}, to_ts is {}'.format(self.word, self.from_ts, self.to_ts)
class Gesture:
    ''' A class to include information about gestures
        In one turn, we can have multiple gestures. These gestures can also be from another speaker (when there is an overlap between speakers)
    '''
    def __init__(self, is_gesture, g_from_ts, g_to_ts, g_type, g_referent, g_comment, g_hand, participant_ID):
        self.is_gesture = is_gesture
        self.g_type = g_type
        self.g_referent = g_referent
        self.g_comment = g_comment
        self.g_hand = g_hand
        self.g_from_ts = g_from_ts
        self.g_to_ts = g_to_ts
        self.participant_ID = participant_ID