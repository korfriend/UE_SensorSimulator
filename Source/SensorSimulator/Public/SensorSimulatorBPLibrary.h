// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"

#include "Engine/Engine.h"
#include "Components/StaticMeshComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "LidarPointCloud.h"
#include "LidarPointCloudComponent.h"

#include "SensorSimulatorBPLibrary.generated.h"

/* 
*	Function library class.
*	Each function in it is expected to be static and represents blueprint node that can be called in any blueprint.
*
*	When declaring function you can define metadata for the node. Key function specifiers will be BlueprintPure and BlueprintCallable.
*	BlueprintPure - means the function does not affect the owning object in any way and thus creates a node without Exec pins.
*	BlueprintCallable - makes a function which can be executed in Blueprints - Thus it has Exec pins.
*	DisplayName - full name of the node, shown when you mouse over the node and in the blueprint drop down menu.
*				Its lets you name the node using characters not allowed in C++ function names.
*	CompactNodeTitle - the word(s) that appear on the node.
*	Keywords -	the list of keywords that helps you to find node when you search for it using Blueprint drop-down menu. 
*				Good example is "Print String" node which you can find also by using keyword "log".
*	Category -	the category your node will be under in the Blueprint drop-down menu.
*
*	For more info on custom blueprint nodes visit documentation:
*	https://wiki.unrealengine.com/Custom_Blueprint_Node_Creation
*/

USTRUCT(BlueprintType)
struct FLidarSensorOut
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<float> depthArrayOut;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FColor> colorArrayOut;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FLidarPointCloudPoint> lidarPointsOut;
};

DECLARE_DYNAMIC_DELEGATE_OneParam(FAsyncDelegate, FLidarSensorOut, SensorOut);

UCLASS()
class USensorSimulatorBPLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_UCLASS_BODY()

		UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "LidarScan", Keywords = "LidarSensorSimulator Async Scan"))
		static void LidarSensorAsyncScan4(
			const TArray<FLidarPointCloudPoint>& lidarPoints, ULidarPointCloudComponent* lidarPointCloud, USceneCaptureComponent2D* sceneCapture,
			TArray<FLidarPointCloudPoint>& lidarPointsOut, TArray<float>& depthArrayOut, TArray<FColor>& colorArrayOut,
			FAsyncDelegate Out, const bool asyncScan = true,
			const float vFovSDeg = -15.f, const float vFovEDeg = 15.f, const int lidarChannels = 32, const float hfovDeg = 90.f, const int lidarResolution = 100, const float lidarRange = 1000.f);
};
