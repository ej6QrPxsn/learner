#ifndef COMMON_HPP
#define COMMON_HPP

auto const TARGET_UPDATE = 2500;
auto const BATCH_SIZE = 32;

auto const REPLAY_PERIOD = 40;
const auto TRACE_LENGTH = 80;
const auto SEQ_LENGTH = 1 + REPLAY_PERIOD + TRACE_LENGTH;
const auto REPLAY_BUFFER_MIN_SIZE = 2500;
const auto REPLAY_BUFFER_ADD_PRINT_SIZE = 500;

#endif // COMMON_HPP
