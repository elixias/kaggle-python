### Here's a code that can be used to calculate the score for blackjack_hand_greater_than
"""
"""

def maximise_a(currpts, a):
    """Gets the immediate winning score with the most 'A's
    """
    for i in reversed(range(a+1)):
        a_pts = (i)*11 + (a-i)
        if ( a_pts + currpts ) <= 21:
            return a_pts + currpts
    return a + currpts
    

def calc_points(hand):
    hand = [ 10 if card in ['J','Q','K'] else card for card in hand ] ## replaces J, Q, K with flat scores
    currpts = sum([int(h) for h in hand if h is not 'A'])             ## sum those up                           
    score = maximise_a(currpts,hand.count('A'))                       ## gets the best score that maximises a
    return score if score <= 21 else 0

def blackjack_hand_greater_than(hand_1, hand_2):
	"""blackjack_hand_greater_than(['A','K','9'],['J','9'])
	"""
    return calc_points(hand_1) > calc_points(hand_2)