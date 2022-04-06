/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "DetectionResult.hpp"
#include "DetectorPostProcessing.hpp"
#include "hal.h"
#include "YoloFastestModel.hpp"
#include "UseCaseHandler.hpp"
#include "UseCaseCommonUtils.hpp"

#include <catch.hpp>

TEST_CASE("Model info")
{
    /* Model wrapper object. */
    arm::app::YoloFastestModel model;

    /* Load the model. */
    REQUIRE(model.Init());

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    caseContext.Set<arm::app::Model&>("model", model);

    REQUIRE(model.ShowModelInfoHandler());
}


TEST_CASE("Inference by index")
{
    hal_platform    platform;
    platform_timer  timer;

    /* Initialise the HAL and platform. */
    hal_init(&platform, &timer);
    hal_platform_init(&platform);

    /* Model wrapper object. */
    arm::app::YoloFastestModel model;

    /* Load the model. */
    REQUIRE(model.Init());

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{&platform, "object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<hal_platform&>("platform", platform);
    caseContext.Set<arm::app::Model&>("model", model);
    caseContext.Set<uint32_t>("imgIndex", 0);
    arm::app::object_detection::DetectorPostprocessing postp;
    caseContext.Set<arm::app::object_detection::DetectorPostprocessing&>("postprocess", postp);

    REQUIRE(arm::app::ObjectDetectionHandler(caseContext, 0, false));
}


TEST_CASE("Inference run all images")
{
    hal_platform    platform;
    platform_timer  timer;

    /* Initialise the HAL and platform. */
    hal_init(&platform, &timer);
    hal_platform_init(&platform);

    /* Model wrapper object. */
    arm::app::YoloFastestModel model;

    /* Load the model. */
    REQUIRE(model.Init());

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{&platform, "object_detection"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<hal_platform&>("platform", platform);
    caseContext.Set<arm::app::Model&>("model", model);
    caseContext.Set<uint32_t>("imgIndex", 0);
    arm::app::object_detection::DetectorPostprocessing postp;
    caseContext.Set<arm::app::object_detection::DetectorPostprocessing&>("postprocess", postp);

    REQUIRE(arm::app::ObjectDetectionHandler(caseContext, 0, true));
}


TEST_CASE("List all images")
{
    hal_platform    platform;
    platform_timer  timer;

    /* Initialise the HAL and platform. */
    hal_init(&platform, &timer);
    hal_platform_init(&platform);

    /* Model wrapper object. */
    arm::app::YoloFastestModel model;

    /* Load the model. */
    REQUIRE(model.Init());

    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    caseContext.Set<hal_platform&>("platform", platform);
    caseContext.Set<arm::app::Model&>("model", model);

    REQUIRE(arm::app::ListFilesHandler(caseContext));
}
