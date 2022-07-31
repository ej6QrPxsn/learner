#ifndef COMMON_HPP
#define COMMON_HPP

const auto TARGET_UPDATE = 2500;
const auto BATCH_SIZE = 64;
const auto LEARNING_RATE = 1e-4;
const auto EPSILON = 1e-3;

const auto REPLAY_PERIOD = 40;
const auto TRACE_LENGTH = 80;
const auto SEQ_LENGTH = 1 + REPLAY_PERIOD + TRACE_LENGTH;

const auto MAX_REPLAY_QUEUE_SIZE = 128;
const auto REPLAY_BUFFER_ADD_PRINT_SIZE = 500;
const auto REPLAY_BUFFER_MIN_SIZE = 2500;
const auto RETURN_TRANSITION_SIZE = 32;
const auto REPLAY_BUFFER_SIZE = 25000;

const auto RETRACE_LAMBDA = 0.95;
const auto RESCALING_EPSILON = 1e-3;
const auto ETA = 0.9;
const auto DISCOUNT_GAMMA = 0.997;

const auto HIGH_REWARD_BUFFER_SIZE = 100;
const auto HIGH_REWARD_SIZE = 10;
const auto HIGH_REWARD_RATIO = 1. / 256.;
const auto REPLAY = 1;
const auto HIGH_REWARD = 2;

const auto ACTION_SIZE = 9;

#endif // COMMON_HPP
