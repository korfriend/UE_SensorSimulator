// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include <Containers/Array.h>

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

// https://forums.unrealengine.com/t/ufunction-in-ustruct/354167/4
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
USTRUCT(BlueprintType)
struct FLidarSensorOut360
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<float> depthArrayOut360;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FColor> colorArrayOutF;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FColor> colorArrayOutL;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FColor> colorArrayOutB;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FColor> colorArrayOutR;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FLidarPointCloudPoint> lidarPointsOut360;
};
USTRUCT(BlueprintType)
struct FBytes
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<uint8> ArrayOut;
};

USTRUCT(BlueprintType)
struct FTArrayBytes 
{
	GENERATED_BODY()

public:
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
		TArray<FBytes> FArrayOut;
};

DECLARE_DYNAMIC_DELEGATE_OneParam(FAsyncDelegate, FLidarSensorOut, SensorOut);
DECLARE_DYNAMIC_DELEGATE_OneParam(FAsyncDelegate360, FLidarSensorOut360, SensorOut360);

UCLASS()

class USensorSimulatorBPLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_UCLASS_BODY()

	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "LidarScan", Keywords = "LidarSensorSimulator Async Scan"))
	static void LidarSensorAsyncScan4(
		const TArray<FLidarPointCloudPoint>& lidarPoints, ULidarPointCloudComponent* lidarPointCloud, USceneCaptureComponent2D* sceneCapture,
		FAsyncDelegate Out, FLidarSensorOut& sensorOut, const bool asyncScan = true,
		const float vFovSDeg = -15.f, const float vFovEDeg = 15.f, const int lidarChannels = 32, const float hfovDeg = 90.f, const int lidarResolution = 100, const float lidarRange = 1000.f);

	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "ReadSemantic", Keywords = "Read Semantic"))
	static void ReadSemantic(USceneCaptureComponent2D* sceneCapture, TArray<FColor>& semanticArrayOut);
	
	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "SensorOutToBytes", Keywords = "Sensor Out to Bytes Array"))
	static void SensorOutToBytes(const TArray<FLidarSensorOut>& lidarSensorOuts, TArray <FBytes>& bytePackets, FString& bytesInfo, int& bytesPoints, int& bytesColorMap, int& bytesDepthMap, const int packetBytes = 10000);

	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "IntToBytes", Keywords = "Sensor Out to Bytes Array"))
	static void IntToBytes(const int fromInt, TArray <uint8>& bytes);



	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "GetViewProTectionMatrix", Keywords = "GetViewProTectionMatrix"))
		static void GetViewProTectionMatrix(const TArray<USceneCaptureComponent2D*> sceneCapture,
			TArray<FMatrix>& viewprojectionMatrix, TArray<FMatrix>& projectionMatrix, TArray<FMatrix>& viewMatrix);
	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "PosAndRotToBytes", Keywords = "camera Position AndRotation ToBytes"))
		static void PosAndRotToBytes(const FVector position, const FRotator rotation, TArray<uint8>& bytePackets);
	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "ConvertBytesToVectorAndRotator", Keywords = "for test PosAndRotToBytes "))
		static void ConvertBytesToVectorAndRotator(const TArray<uint8>& InBytes, FVector& OutVector, FRotator& OutRotator);
	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "CamInfoToBytes", Keywords = "Cam Matrix Array ,fov, aspect ratio  to Bytes Array"))
		static void	CamInfoToBytes(const TArray<FMatrix>& camMat, const int fov, const float aspectRatio, TArray<uint8>& bytePackets);

	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "LidarScan360", Keywords = "LidarSensorSimulator Async with scenecap as input"))
		static void	LidarSensorAsyncScan360(const TArray<FLidarPointCloudPoint>& lidarPoints,
			ULidarPointCloudComponent* lidarPointCloud, 
			USceneCaptureComponent2D* sceneCaptureF, USceneCaptureComponent2D* sceneCaptureL,
			USceneCaptureComponent2D* sceneCaptureB, USceneCaptureComponent2D* sceneCaptureR,
			FAsyncDelegate360 Out, FLidarSensorOut360& sensorOut, const bool asyncScan = true,
			const float vFovSDeg = -15.f, const float vFovEDeg = 15.f, const int lidarChannels = 32, const float hfovDeg = 90.f, const int lidarResolution = 100, const float lidarRange = 1000.f);
	
	UFUNCTION(BlueprintCallable, Category = "SensorSimulator", meta = (DisplayName = "SensorOutToBytes360", Keywords = "SensorOut360 to Bytes Array"))
		static void SensorOutToBytes360(const FLidarSensorOut360& lidarSensorOuts,
			//TArray<FBytes>& bytePackets,
			FTArrayBytes& bytePackets,
			FString& bytesInfo, int& bytesPoints, int& bytesColorMap,
			int& bytesDepthMap, const int packetBytes);

};

