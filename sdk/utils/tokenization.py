import os
import jieba
import collections
import unicodedata
import six
import sys
import re
import glob
import pickle
import tensorflow as tf
from collections import defaultdict
# import logging

# Set contains all letter and number characters.
_ALPHANUMERIC_CHAR_SET = set(
    six.unichr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(six.unichr(i)).startswith("L") or
        unicodedata.category(six.unichr(i)).startswith("N")))

_ESCAPE_CHARS = set(u"\\_u;0123456789")

PAD = '<pad>' # pad_id: 0
UNK = '<unk>' # unk_id: 1
SOS = '<s>' # sos_id:2
EOS = '</s>' # eos_id:3
NUM = '<num>' # num_id:4
SEP = '<sep>' # sep:

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
NUM_ID = 4
SEP_ID = 5

RESERVED_TOKENS = [PAD,UNK,SOS,EOS,NUM,SEP]


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


# def split_text(text):
#   """Runs basic whitespace cleaning and splitting on a piece of text.
#   可以优化
#   """
#   text = text.strip()
#   if not text:
#     return []
#   tokens = jieba.lcut(text)
#   # print(tokens)
#   return tokens

def _escape_token(token, alphabet):
  r"""Replace characters that aren't in the alphabet and append "_" to token.

  Apply three transformations to the token:
    1. Replace underline character "_" with "\u", and backslash "\" with "\\".
    2. Replace characters outside of the alphabet with "\###;", where ### is the
       character's Unicode code point.
    3. Appends "_" to mark the end of a token.

  Args:
    token: unicode string to be escaped
    alphabet: list of all known characters

  Returns:
    escaped string
  """
  token = token.replace(u"\\", u"\\\\").replace(u"_", u"\\u")
  ret = [c for c in token]
  return u"".join(ret) + "_"


def _split_token_to_subtokens(token, subtoken_dict, max_subtoken_length=4):
  """Splits a token into subtokens defined in the subtoken dict."""
  ret = []
  start = 0

  token_len = len(token)

  while start < token_len:
  # Find the longest subtoken, so iterate backwards.
    for end in range(min(token_len, start + max_subtoken_length), start, -1):
      subtoken = token[start:end]
      if subtoken in subtoken_dict:
        ret.append(subtoken)
        start = end
        break
    else:  # Did not break
      # If there is no possible encoding of the escaped token then one of the
      # characters in the token is not in the alphabet. This should be
      # impossible and would be indicative of a bug.
      ret.append(UNK)
      start = end
      # raise ValueError("Was unable to split token \"%s\" into subtokens." %
                       # token)
  return ret

def _filter_and_bucket_subtokens(subtoken_counts, min_count):
  """Return a bucketed list of subtokens that are filtered by count.

  Args:
    subtoken_counts: defaultdict mapping subtokens to their counts
    min_count: int count used to filter subtokens

  Returns:
    List of subtoken sets, where subtokens in set i have the same length=i.
  """
  # Create list of buckets, where subtokens in bucket i have length i.
  subtoken_buckets = []
  for subtoken, count in six.iteritems(subtoken_counts):
    if count < min_count:  # Filter out subtokens that don't appear enough
      continue
    while len(subtoken_buckets) <= len(subtoken):
      subtoken_buckets.append(set())
    subtoken_buckets[len(subtoken)].add(subtoken)
  return subtoken_buckets


def _gen_new_subtoken_list(
    subtoken_counts, min_count, alphabet, reserved_tokens):
  """Generate candidate subtokens ordered by count, and new max subtoken length.

  Add subtokens to the candiate list in order of length (longest subtokens
  first). When a subtoken is added, the counts of each of its prefixes are
  decreased. Prefixes that don't appear much outside the subtoken are not added
  to the candidate list.

  For example:
    subtoken being added to candidate list: 'translate'
    subtoken_counts: {'translate':10, 't':40, 'tr':16, 'tra':12, ...}
    min_count: 5

  When 'translate' is added, subtoken_counts is updated to:
    {'translate':0, 't':30, 'tr':6, 'tra': 2, ...}

  The subtoken 'tra' will not be added to the candidate list, because it appears
  twice (less than min_count) outside of 'translate'.

  Args:
    subtoken_counts: defaultdict mapping str subtokens to int counts
    min_count: int minumum count requirement for subtokens
    alphabet: set of characters. Each character is added to the subtoken list to
      guarantee that all tokens can be encoded.
    reserved_tokens: list of tokens that will be added to the beginning of the
      returned subtoken list.

  Returns:
    List of candidate subtokens in decreasing count order, and maximum subtoken
    length
  """

  # Create a list of (count, subtoken) for each candidate subtoken.
  subtoken_candidates = []

  # Use bucketted list to iterate through subtokens in order of length.
  # subtoken_buckets[i] = set(subtokens), where each subtoken has length i.
  subtoken_buckets = _filter_and_bucket_subtokens(subtoken_counts, min_count)
  max_subtoken_length = len(subtoken_buckets) - 1

  # Go through the list in reverse order to consider longer subtokens first.
  for subtoken_len in range(max_subtoken_length, 0, -1):
    for subtoken in subtoken_buckets[subtoken_len]:
      count = subtoken_counts[subtoken]

      # Possible if this subtoken is a prefix of another token.
      if count < min_count:
        continue

      # Ignore alphabet/reserved tokens, which will be added manually later.
      if subtoken not in alphabet and subtoken not in reserved_tokens:
        subtoken_candidates.append((count, subtoken))

      # Decrement count of the subtoken's prefixes (if a longer subtoken is
      # added, its prefixes lose priority to be added).
      for end in range(1, subtoken_len):
        subtoken_counts[subtoken[:end]] -= count

  # Add alphabet subtokens (guarantees that all strings are encodable).
  subtoken_candidates.extend((subtoken_counts.get(a, 0), a) for a in alphabet)

  # Order subtoken candidates by decreasing count.
  subtoken_list = [t for _, t in sorted(subtoken_candidates, reverse=True)]

  # Add reserved tokens to beginning of the list.
  subtoken_list = reserved_tokens + subtoken_list
  return subtoken_list, max_subtoken_length



def _count_and_gen_subtokens(
    token_counts, alphabet, subtoken_dict, max_subtoken_length=4):
  """Count number of times subtokens appear, and generate new subtokens.

  Args:
    token_counts: dict mapping tokens to the number of times they appear in the
      original files.
    alphabet: list of allowed characters. Used to escape the tokens, which
      guarantees that all tokens can be split into subtokens.
    subtoken_dict: dict mapping subtokens to ids.
    max_subtoken_length: maximum length of subtoken in subtoken_dict.

  Returns:
    A defaultdict mapping subtokens to the number of times they appear in the
    tokens. The dict may contain new subtokens.
  """
  subtoken_counts = collections.defaultdict(int)
  for token, count in six.iteritems(token_counts):
    token = _escape_token(token, alphabet)
    subtokens = _split_token_to_subtokens(
        token, subtoken_dict, max_subtoken_length)

    # Generate new subtokens by taking substrings from token.
    start = 0
    for subtoken in subtokens:
      for end in range(start + 1, len(token) + 1):
        new_subtoken = token[start:end]
        subtoken_counts[new_subtoken] += count
      start += len(subtoken)
  return subtoken_counts

def _generate_alphabet_dict(iterable):
  """Create set of characters that appear in any element in the iterable."""
  alphabet = {c for token in iterable for c in token}
  alphabet |= _ESCAPE_CHARS  # Add escape characters to alphabet set.
  return alphabet

def _generate_subtokens(token_counts, alphabet, min_count,reserved_tokens,max_subtoken_length = 2,num_iterations=4):
    """Create a list of subtokens in decreasing order of frequency.
    Args:
        token_counts: dict mapping str tokens -> int count
        alphabet: set of characters
        min_count: int minimum number of times a subtoken must appear before it is
        added to the vocabulary.
        num_iterations: int number of iterations to generate new tokens.
        reserved_tokens: list of tokens that will be added to the beginning to the
        returned subtoken list.

    Returns:
        Sorted list of subtokens (most frequent first)
    """
    # Use alphabet set to create initial list of subtokens
    subtoken_list = list(alphabet)
    
    # On each iteration, segment all words using the subtokens defined in
    # subtoken_dict, count how often the resulting subtokens appear, and update
    # the dictionary with subtokens w/ high enough counts.
    for i in range(num_iterations):
        # Generate new subtoken->id dictionary using the new subtoken list.
        subtoken_dict = _list_to_index_dict(subtoken_list)

        # Create dict mapping subtoken->count, with additional subtokens created
        # from substrings taken from the tokens.
        subtoken_counts = _count_and_gen_subtokens(
                token_counts, alphabet, subtoken_dict, max_subtoken_length)

        # Generate new list of subtokens sorted by subtoken count.
        subtoken_list, max_subtoken_length = _gen_new_subtoken_list(
                subtoken_counts, min_count, alphabet, reserved_tokens)

    return subtoken_list

def _count_tokens(files, file_byte_limit=1e6):
  """Return token counts of words in the files.

  Samples file_byte_limit bytes from each file, and counts the words that appear
  in the samples. The samples are semi-evenly distributed across the file.

  Args:
    files: List of filepaths
    file_byte_limit: Max number of bytes that will be read from each file.

  Returns:
    Dictionary mapping tokens to the number of times they appear in the sampled
    lines from the files.
  """
  token_counts = collections.defaultdict(int)

  for filepath in files:
    with tf.gfile.Open(filepath, mode="r") as reader:
      file_byte_budget = file_byte_limit
      counter = 0
      lines_to_skip = int(reader.size() / (file_byte_budget * 2))
      for line in reader:
        if counter < lines_to_skip:
          counter += 1
        else:
          if file_byte_budget < 0:
            break
          line = line.strip()
          file_byte_budget -= len(line)
          counter = 0

          # Add words to token counts
          for token in whitespace_tokenize(line):
            token_counts[token] += 1
  return token_counts



def _list_to_index_dict(lst):
  """Create dictionary mapping list items to their indices in the list."""
  return {item: n for n, item in enumerate(lst)}


class Tokenizer:
  def __init__(self,
              params):
    self.params = params
    
    self.vocab_file = self.params['vocab_file']
    extra_reserved_tokens = self.params['extra_reserved_tokens']
    print('extra_reserved_tokens:{}'.format(extra_reserved_tokens))
    if extra_reserved_tokens:
      RESERVED_TOKENS.extend(extra_reserved_tokens)    
    self.reserved_tokens = RESERVED_TOKENS

    if self.params['update_vocab'] or (not os.path.isfile(self.vocab_file)):
      print('start to generate vocab')
      min_count = self.params['min_count']
      self.subtoken_list = self.make_vocab(min_count)
    else:
      self.subtoken_list = [subtoken.strip() for subtoken in open(self.vocab_file,'r').readlines()]
    self.subtoken_to_id_dict = _list_to_index_dict(self.subtoken_list)
    assert self.subtoken_to_id_dict[PAD]==PAD_ID
    assert self.subtoken_to_id_dict[UNK]==UNK_ID
    assert self.subtoken_to_id_dict[SEP]==SEP_ID
    assert self.subtoken_to_id_dict[NUM]==NUM_ID
    assert self.subtoken_to_id_dict[SOS]==SOS_ID
    assert self.subtoken_to_id_dict[EOS]==EOS_ID

    self.alphabet = _generate_alphabet_dict(self.subtoken_list)
    self.vocab_size = len(self.subtoken_list)

  def encode(self,raw_string,padding=True,start_mark=False,end_mark=False):
    """Encodes a string into a list of int subtoken ids."""
    ret = []
    tokens = whitespace_tokenize(raw_string)
    # print(tokens)

    for token in tokens:
      ret.extend(self._token_to_subtoken_ids(token))
    ret_len = len(ret)
    ret_padding_len = ret_len
    if start_mark:
      ret = [SOS_ID] + ret
      ret_padding_len+=1
    if end_mark:
      ret = ret + [EOS_ID]
      ret_padding_len+=1
    if padding:
      ret_len+=1
      padding_len = self.params['max_length']
      if ret_padding_len > padding_len:
        return None,None
        # print(raw_string,ret_padding_len)
        # raise 'padding sentence longer than maximum len'
      else:
        ret.extend([PAD_ID]*(padding_len-ret_padding_len))
    return ret,ret_len

  def decode(self,subtokens):
    """Converts list of int subtokens ids into a string."""
    if not subtokens:
      return ""

    assert isinstance(subtokens, list) and isinstance(subtokens[0], int), (
        "Subtokens argument passed into decode() must be a list of integers.")
    res = ''.join(self._subtoken_ids_to_tokens(subtokens))
    res=res.replace('<sep>',' ')
    res=res.split('</s>')[0] + '</s>'
    res=res.replace('<pad>','')
    return res

  def _token_to_subtoken_ids(self,token):
    _cache_size = 2 ** 20
    _cache = [(None, None)] * _cache_size

    """Encode a single token into a list of subtoken ids."""
    cache_location = hash(token) % _cache_size
    cache_key, cache_value = _cache[cache_location]
    if cache_key == token:
      return cache_value


    ret = _split_token_to_subtokens(
        _escape_token(token, self.alphabet), self.subtoken_to_id_dict)
    ret = [self.subtoken_to_id_dict[subtoken_id] for subtoken_id in ret]

    _cache[cache_location] = (token, ret)
    return ret
  def _subtoken_ids_to_tokens(self,subtokens):
    """Convert list of int subtoken ids to a list of string tokens."""
    escaped_tokens = "".join([
        self.subtoken_list[s] for s in subtokens
        if s < len(self.subtoken_list)])
    escaped_tokens = escaped_tokens.split("_")

    # All tokens in the vocabulary list have been escaped (see _escape_token())
    # so each token must be unescaped when decoding.
    ret = []
    for token in escaped_tokens:
      if token:
        ret.append(token)
    return ret
  def make_vocab(self,min_count=4):

    datafiles=[self.params['src_file']]
    if not self.params['is_tgt_label']:
      datafiles.append(self.params['tgt_file'])
    print('vocab from: {}'.format(datafiles))
    token_counts=_count_tokens(files=datafiles)
    print('token_counts len: {}'.format(len(token_counts)))
    alphabet = _generate_alphabet_dict(token_counts)
    print('alphabet: {}'.format(len(alphabet)))
    subtoken_list = _generate_subtokens(
        token_counts, alphabet, min_count,self.reserved_tokens,max_subtoken_length = 2,num_iterations=8)
    # logger.info('vocab_size: {}'.format(len(subtoken_list)))
    print('vocab size:{}'.format(len(subtoken_list)))

    with open(self.vocab_file,'w') as f:
        for token in subtoken_list:
          f.write(token+'\n')
    return subtoken_list
