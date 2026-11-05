//===-- OriginOmpFileBridge.h - Origin ompfile_mpp_* bridges ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The seam between the MPI plugin core (rtl.cpp) and the origin rank's
// libompfile bridges (OriginOmpFileBridge.cpp: the extern "C" ompfile_mpp_*
// entrypoints of ompfile_mpp_abi.h and the origin handle table).
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_MPI_ORIGIN_OMPFILE_BRIDGE_H
#define OFFLOAD_PLUGINS_NEXTGEN_MPI_ORIGIN_OMPFILE_BRIDGE_H

#include "EventSystem.h"

#include "llvm/Support/Error.h"

namespace llvm::omp::target::plugin {

struct MPIPluginTy;

// OriginOmpFileBridge.cpp: the plugin the origin ompfile_mpp_* bridges use.
// rtl.cpp calls both, and that call is what pulls the bridge object out of the
// static omptarget.rtl.mpi archive.
void ompfilePublishMPIPlugin(MPIPluginTy *Plugin);
void ompfileRetractMPIPlugin(MPIPluginTy *Plugin);

// Defined in rtl.cpp, where MPIPluginTy is complete.
llvm::Error ompfileInitMPIPlugin(MPIPluginTy &Plugin);
bool ompfileEnsureEventSystem(MPIPluginTy &Plugin);
EventSystemTy &ompfileEventSystem(MPIPluginTy &Plugin);
void ompfileDiagLog(const char *Phase, int Rank, int DeviceCount, int DeviceId,
                    int ExtraA, int ExtraB);

} // namespace llvm::omp::target::plugin

#endif // OFFLOAD_PLUGINS_NEXTGEN_MPI_ORIGIN_OMPFILE_BRIDGE_H
