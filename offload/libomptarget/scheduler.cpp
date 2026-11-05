//===---------------- scheduler.cpp - Target-Task scheduler ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Scheduler used to manage multiple devices of the same type.
//
//===----------------------------------------------------------------------===//

#include "scheduler.h"
#include "PluginManager.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

// Scheduler interface in libomptarget
// =============================================================================

int Schedule(ident_t *Loc, int DeviceId) {
  DP("[SCHEDULER] Asked to schedule for Device%d (start)\n", DeviceId);

  const int32_t TaskId = __kmpc_target_task_current_task_id();

  // A concrete device selected outside an OpenMP target task must be
  // preserved. The OMPC scheduler is responsible for assigning devices to
  // target tasks, not for overriding an explicit synchronous device choice.
  if (DeviceId >= 0 && TaskId == -1)
    return DeviceId;

  int ScheduledDevId = -1;
  bool DMDASOwned = false;

  if (__kmpc_dmdas_get_task_device) {
    ScheduledDevId = __kmpc_dmdas_get_task_device(TaskId);

    if (ScheduledDevId >= 0)
      DMDASOwned = true;
  }

  if (!DMDASOwned) {

    ScheduledDevId = __kmpc_scheduleTask(TaskId);
  }

  // No scheduling decision for this task (it is not part of a recorded
  // graph): honour an explicit device() clause instead of collapsing every
  // such target region onto device 0, and use device 0 only without one.
  if (ScheduledDevId < 0)
    ScheduledDevId = DeviceId >= 0 ? DeviceId : 0;

  if (TaskId != -1) {

    if (!DMDASOwned) {
      __kmpc_setTaskState(TaskId, TaskState::DISPATCHED);
    }

    if (__kmpc_dmdas_target_dispatched)
      __kmpc_dmdas_target_dispatched(TaskId);
  }

  return ScheduledDevId;
}

void PinTask(ident_t *Loc, int DeviceId) {
  const int32_t TaskId = __kmpc_target_task_current_task_id();
  if (TaskId != -1) {
    __kmpc_setTaskState(TaskId, TaskState::DISPATCHED);
    __kmpc_pinTask(TaskId, DeviceId);
    DP("[SCHEDULER] Pinning task %d at device %d!\n", TaskId, DeviceId);
  }
}

void MarkExecutedTask() {
  const int32_t TaskId = __kmpc_target_task_current_task_id();

  if (TaskId == -1)
    return;

  bool DMDASOwned = false;

  if (__kmpc_dmdas_get_task_device) {
    const int DMDASDevice = __kmpc_dmdas_get_task_device(TaskId);

    DMDASOwned = DMDASDevice >= 0;
  }

  if (!DMDASOwned) {
    __kmpc_setTaskState(TaskId, TaskState::EXECUTED);
  }
}

void MarkExecutedPinnedTask(const DeviceTy &Device) {
  const int32_t TaskId = __kmpc_target_task_current_task_id();

  if (TaskId == -1)
    return;

  bool DMDASOwned = false;

  if (__kmpc_dmdas_get_task_device) {
    const int DMDASDevice = __kmpc_dmdas_get_task_device(TaskId);

    DMDASOwned = DMDASDevice >= 0;
  }

  if (!DMDASOwned) {
    __kmpc_setTaskState(TaskId, TaskState::EXECUTED);
  }
}

int32_t GetCurrentTargetTaskID() {
  return __kmpc_target_task_current_task_id();
}
