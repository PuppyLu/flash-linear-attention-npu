/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "../chunk_kda_fwd_prepare_tiling_processor.h"
#include "../chunk_kda_fwd_prepare_output_mask.h"

#include <cstdint>
#include <limits>

namespace {

bool CheckOutputMaskContract()
{
    using namespace optiling;
    constexpr uint32_t expectedRequired =
        PREPARE_OUTPUT_MASK_GK | PREPARE_OUTPUT_MASK_AQK |
        PREPARE_OUTPUT_MASK_W | PREPARE_OUTPUT_MASK_U |
        PREPARE_OUTPUT_MASK_KG | PREPARE_OUTPUT_MASK_QG_SCALED;
    constexpr uint32_t expectedOptional =
        PREPARE_OUTPUT_MASK_AKK | PREPARE_OUTPUT_MASK_QG |
        PREPARE_OUTPUT_MASK_Q_HAT | PREPARE_OUTPUT_MASK_K_HAT |
        PREPARE_OUTPUT_MASK_Q_RSTD | PREPARE_OUTPUT_MASK_K_RSTD |
        PREPARE_OUTPUT_MASK_BETA_EFF;
    constexpr uint32_t expectedRecompute =
        expectedRequired | (expectedOptional & ~PREPARE_OUTPUT_MASK_QG);
    constexpr uint32_t expectedSave = expectedRequired | expectedOptional;
    return PREPARE_OUTPUT_COUNT == 13 &&
        PREPARE_REQUIRED_OUTPUT_MASK == expectedRequired &&
        PREPARE_OPTIONAL_OUTPUT_MASK == expectedOptional &&
        PREPARE_RECOMPUTE_OUTPUT_MASK == expectedRecompute &&
        PREPARE_SAVE_OUTPUT_MASK == expectedSave;
}

bool CheckSingleElementOptionalOutputBits()
{
    using namespace optiling;
    // q_rstd/k_rstd/beta_eff 允许 shape 的元素数为 1，不能据此清除输出位。
    constexpr uint32_t scalarOutputBits =
        PREPARE_OUTPUT_MASK_Q_RSTD | PREPARE_OUTPUT_MASK_K_RSTD |
        PREPARE_OUTPUT_MASK_BETA_EFF;
    return (PREPARE_OPTIONAL_OUTPUT_MASK & scalarOutputBits) ==
        scalarOutputBits;
}

bool CheckChunkOnlySchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        2, 4, 8, 16, 0, 24, 1024, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 32 && schedule.totalWorkItems == 64 &&
        schedule.usedCoreNum == 24 && schedule.headsPerPartition == 4 &&
        schedule.workspaceBytes ==
            1024 + 24 *
                optiling::CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES;
}

bool CheckArchSpecificWorkspace()
{
    optiling::ChunkKdaFwdPrepareScheduleContext arch22Context{
        2, 4, 8, 16, 0, 24, 1024, false};
    auto arch35Context = arch22Context;
    arch35Context.workspaceSlotBytes =
        optiling::CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES;
    optiling::ChunkKdaFwdPrepareSchedule arch22Schedule;
    optiling::ChunkKdaFwdPrepareSchedule arch35Schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(arch22Context)
               .Process(arch22Schedule) &&
        optiling::ChunkKdaFwdPrepareTilingProcessor(arch35Context)
            .Process(arch35Schedule) &&
        arch22Schedule.workspaceBytes ==
            1024 + 24 *
                optiling::CHUNK_KDA_FWD_PREPARE_ARCH22_WORKGROUP_BYTES &&
        arch35Schedule.workspaceBytes ==
            1024 + 24 *
                optiling::CHUNK_KDA_FWD_PREPARE_ARCH35_WORKGROUP_BYTES;
}

bool CheckGvaHeadSplitSchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 4, 8, 1, 0, 24, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 1 && schedule.totalWorkItems == 4 &&
        schedule.usedCoreNum == 4 && schedule.headsPerPartition == 2;
}

bool CheckVarLenSchedule()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 3, 12, 0, 7, 6, 4096, true};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule) &&
        schedule.chunkWorkItems == 7 && schedule.totalWorkItems == 21 &&
        schedule.usedCoreNum == 6 && schedule.headsPerPartition == 4;
}

bool CheckInvalidGva()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 3, 8, 1, 0, 24, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return !optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule);
}

bool CheckChunkWorkItemOverflow()
{
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        1, 1, 2, 0, 24, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return !optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule);
}

bool CheckHeadPartitionWorkItemOverflow()
{
    const uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    const optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 3, 3, maxUint32 / 2 + 1, 0, maxUint32, 0, false};
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    return !optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule);
}

bool CheckArch22TailBalance()
{
    using namespace optiling;
    auto process = [](const ChunkKdaFwdPrepareScheduleContext &c,
                      uint64_t work, uint32_t heads, uint32_t cores) {
        ChunkKdaFwdPrepareSchedule s;
        return ChunkKdaFwdPrepareTilingProcessor(c).Process(s) &&
               s.totalWorkItems == work && s.headsPerPartition == heads &&
               s.usedCoreNum == cores;
    };
    ChunkKdaFwdPrepareScheduleContext c{1, 96, 96, 64, 0, 20, 0, false};
    if (!process(c, 1536, 4, 20)) return false; // dense
    c.isVarLen = true; c.totalVarLenChunks = 64;
    if (!process(c, 1536, 4, 20)) return false; // varlen
    c.isVarLen = false; c.chunksPerSequence = 60;
    if (!process(c, 60, 96, 20)) return false; // full waves
    c.chunksPerSequence = 64; c.qkHeadNum = 32;
    if (!process(c, 64, 96, 20)) return false; // GVA group 3
    c.qkHeadNum = 24;
    if (!process(c, 1536, 4, 20)) return false; // GVA group 4
    c.qkHeadNum = 7; c.valueHeadNum = 7;
    if (!process(c, 128, 4, 20)) return false; // final 3 heads
    c.qkHeadNum = 96; c.valueHeadNum = 96;
    c.workspaceSlotBytes = CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES;
    if (!process(c, 64, 96, 20)) return false; // arch35 unchanged
    c.workspaceSlotBytes = CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES;
    c.isVarLen = true;
    c.totalVarLenChunks = std::numeric_limits<uint32_t>::max();
    if (!process(c, std::numeric_limits<uint32_t>::max(), 96, 20))
        return false; // partition overflow falls back
    return true;
}

} // namespace

int main()
{
    return CheckOutputMaskContract() && CheckSingleElementOptionalOutputBits() &&
                   CheckChunkOnlySchedule() &&
                   CheckArchSpecificWorkspace() &&
                   CheckGvaHeadSplitSchedule() &&
                   CheckVarLenSchedule() && CheckInvalidGva() &&
                   CheckChunkWorkItemOverflow() &&
                   CheckHeadPartitionWorkItemOverflow() &&
                   CheckArch22TailBalance()
               ? 0
               : 1;
}
