// Copyright Epic Games, Inc. All Rights Reserved.

#include "SensorSimulatorBPLibrary.h"
#include "SensorSimulator.h"

#include "Async/Async.h"
#include "Misc/ScopeLock.h"
#include "Kismet/GameplayStatics.h"
#include "Engine/TextureRenderTarget2D.h"

#include <map>

//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"    
//#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"

std::map<USceneCaptureComponent2D*, bool> mapCompleted;

bool ShowOnScreenDebugMessages = true;
//ScreenMsg
FORCEINLINE void ScreenMsg(const FString& Msg)
{
	if (!ShowOnScreenDebugMessages) return;
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, *Msg);
}
FORCEINLINE void ScreenMsg(const FString& Msg, const float Value)
{
	if (!ShowOnScreenDebugMessages) return;
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("%s %f"), *Msg, Value));
}
FORCEINLINE void ScreenMsg(const FString& Msg, const FString& Msg2)
{
	if (!ShowOnScreenDebugMessages) return;
	GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, FString::Printf(TEXT("%s %s"), *Msg, *Msg2));
}

USensorSimulatorBPLibrary::USensorSimulatorBPLibrary(const FObjectInitializer& ObjectInitializer)
: Super(ObjectInitializer)
{

}

auto LidarScan = [](const TArray<FLidarPointCloudPoint>& lidarPoints, ULidarPointCloudComponent* lidarPointCloud, USceneCaptureComponent2D* sceneCapture,
	FAsyncDelegate Out, FLidarSensorOut& sensorOut,
	const float vFovSDeg, const float vFovEDeg, const int lidarChannels, const float hfovDeg, const int lidarResolution, const float lidarRange)
{

	FMinimalViewInfo viewInfo;
	sceneCapture->GetCameraView(0, viewInfo);
	UTextureRenderTarget2D* textureRT = sceneCapture->TextureTarget;
	float texWidth = (float)textureRT->SizeX;
	float texHeight = (float)textureRT->SizeY;

	//sceneCapture->HiddenActors()

	FMatrix ViewMatrix, ProjectionMatrix, ViewProjectionMatrix;
	UGameplayStatics::GetViewProjectionMatrix(viewInfo, ViewMatrix, ProjectionMatrix, ViewProjectionMatrix);

	auto RenderTargetResource = textureRT->GameThread_GetRenderTargetResource();
	
	TArray<FColor> buffer;
	sensorOut.colorArrayOut.Init(FColor(), texWidth * texHeight);
	if (RenderTargetResource) {
		RenderTargetResource->ReadPixels(buffer);

		sensorOut.colorArrayOut = buffer;
		//GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, buffer[256 * 100 + 5].ToString());
		//GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, buffer[256 * 200 + 50].ToString());
		//GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, buffer[256 * 10 + 200].ToString());
		//cv::Mat wrappedImage(RenderTarget->GetSurfaceHeight(), RenderTarget->GetSurfaceWidth(), CV_8UC4,
		//	buffer.GetData());
		//
		//std::string OutputFile(TCHAR_TO_UTF8(*OutputVideoFile));
		//cv::imwrite(OutputFile, wrappedImage);
	}


	//TObjectPtr<UTextureRenderTarget2D> RTA;
	//RTA.

	UWorld* currentWorld = lidarPointCloud->GetWorld();

	if (currentWorld == nullptr) {
		return;
	}

	sensorOut.lidarPointsOut = lidarPoints; // copy data

	FVector posLidarSensorWS = lidarPointCloud->GetComponentLocation();
	FRotator rotLidarSensor2World = lidarPointCloud->GetComponentRotation();

	const float deltaDeg = hfovDeg / (float)lidarResolution;

	FCollisionObjectQueryParams targetObjTypes;
	targetObjTypes.AddObjectTypesToQuery(ECC_WorldStatic);
	targetObjTypes.AddObjectTypesToQuery(ECC_WorldDynamic);
	targetObjTypes.AddObjectTypesToQuery(ECC_PhysicsBody);
	targetObjTypes.AddObjectTypesToQuery(ECC_Vehicle);
	targetObjTypes.AddObjectTypesToQuery(ECC_Destructible);

	sensorOut.depthArrayOut.Init(-1.f, lidarChannels * lidarResolution);

	const float vRotDelta = fabs(vFovEDeg - vFovSDeg) / std::max((lidarChannels - 1), 1);
	for (int chIdx = 0; chIdx < lidarChannels; chIdx++) {

		float vRotDeg = vFovSDeg + (float)chIdx * vRotDelta;
		FVector dirStart = FVector(1.f, 0, 0).RotateAngleAxis(vRotDeg, FVector(0, -1, 0));
		FVector posChannelStart = posLidarSensorWS;// +FVector(0, 0, 1.f) * channelInterval * (float)chIdx;

		for (int x = 0; x < lidarResolution; x++) {
			float rotDeg = deltaDeg * (float)x - hfovDeg * 0.5f;
			FVector vecRot = dirStart.RotateAngleAxis(rotDeg, FVector(0, 0, 1));
			FVector dirVec = rotLidarSensor2World.RotateVector(vecRot);
			FVector posChannelEnd = posChannelStart + dirVec * lidarRange;

			// LineTrace
			FHitResult outHit;
			if (currentWorld->LineTraceSingleByObjectType(outHit, posChannelStart, posChannelEnd, targetObjTypes)) {
				//FVector3f hitPosition = FVector3f(outHit.Location);
				// we need the local space position rather than world space position
				FVector3f hitPosition = FVector3f(vecRot * outHit.Distance);

				//UMeshComponent* meshComponent = outHit.GetActor()->GetComponentByClass(UMeshComponent::StaticClass());
				//meshComponent->GetMaterial(0);
				//GetMaterialFromInternalFaceIndex()

				//UMeshComponent* meshComponent = (UMeshComponent * )outHit.GetComponent();
				//UMaterialInterface* mat = meshComponent->GetMaterial(0);

				// to do //
				if (RenderTargetResource) {
					FVector3f hitPositionWS = FVector3f(outHit.Location);
					FPlane psPlane = ViewProjectionMatrix.TransformFVector4(FVector4(hitPositionWS, 1.f));
					float NormalizedX = (psPlane.X / (psPlane.W * 2.f)) + 0.5f;
					float NormalizedY = 1.f - (psPlane.Y / (psPlane.W * 2.f)) - 0.5f;
					FVector2D ScreenPos = FVector2D(NormalizedX * texWidth, NormalizedY * texHeight);
					int sx = (int)ScreenPos.X;
					int sy = (int)ScreenPos.Y;
					FColor color(0, 255, 0, 255);
					if (sx >= 0 && sx < texWidth && sy >= 0 && sy < texHeight && psPlane.Z / psPlane.W > 0) {
						color = buffer[sx + sy * texWidth];
					}
					//if (psPlane.Z / psPlane.W > 0) {
					//	color = FColor(255, 0, 0, 255);
					//}
					FLidarPointCloudPoint pcPoint(hitPosition, color, true, 0);
					sensorOut.lidarPointsOut.Add(pcPoint);
				}
				else {
					FLidarPointCloudPoint pcPoint(hitPosition);
					sensorOut.lidarPointsOut.Add(pcPoint);
				}

				sensorOut.depthArrayOut[x + chIdx * lidarResolution] = FVector::Dist(posChannelStart, FVector(hitPosition));
			}
		}
	}
	//GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Red, FString::FromInt(lidarPointsOut.Num()));
	lidarPointCloud->GetPointCloud()->SetData(sensorOut.lidarPointsOut);

	mapCompleted[sceneCapture] = true;

	AsyncTask(ENamedThreads::GameThread, [Out, &sensorOut]()
		{
			//if (Out != nullptr) {
			// We execute the delegate along with the param
			if (Out.IsBound()) {
				Out.Execute(sensorOut);
			}
			//}

		}
	);
	//
};

void USensorSimulatorBPLibrary::LidarSensorAsyncScan4(
	const TArray<FLidarPointCloudPoint>& lidarPoints, ULidarPointCloudComponent* lidarPointCloud, USceneCaptureComponent2D* sceneCapture,
	FAsyncDelegate Out, FLidarSensorOut& sensorOut, const bool asyncScan,
	const float vFovSDeg, const float vFovEDeg, const int lidarChannels, const float hfovDeg, const int lidarResolution, const float lidarRange
)
{
	if (mapCompleted.find(sceneCapture) == mapCompleted.end())
		mapCompleted[sceneCapture] = true;
	if (!mapCompleted[sceneCapture]) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("Wait..."));
		return;
	}
	mapCompleted[sceneCapture] = false;

	if (asyncScan) {
		// Schedule a thread
		// Pass in our parameters to the lambda expression
		// note that those parameters from the main thread can be released out during this async process
		// pointer parameters are safe because they are conservative
		AsyncTask(ENamedThreads::GameThread, [&lidarPoints, lidarPointCloud, sceneCapture, Out, &sensorOut, asyncScan, vFovSDeg, vFovEDeg, lidarChannels, hfovDeg, lidarResolution, lidarRange]() { // AnyHiPriThreadNormalTask
			LidarScan(lidarPoints, lidarPointCloud, sceneCapture, Out, sensorOut,
				vFovSDeg, vFovEDeg, lidarChannels, hfovDeg, lidarResolution, lidarRange);
			});
	}
	else {
		LidarScan(lidarPoints, lidarPointCloud, sceneCapture, Out, sensorOut,
			vFovSDeg, vFovEDeg, lidarChannels, hfovDeg, lidarResolution, lidarRange);
	}
}


void USensorSimulatorBPLibrary::SensorOutToBytes(const TArray<FLidarSensorOut>& lidarSensorOuts, TArray<uint8>& bytes, FString& bytesInfo)
{
	// compute array size
	uint32 bytesCount = 0;
	int index = 0;
	for (const FLidarSensorOut& sensorOut : lidarSensorOuts) {
		if (sensorOut.lidarPointsOut.Num() > 0) {
			bytesCount += sensorOut.lidarPointsOut.Num() * 4 * 4;
			bytesInfo += FString("Point") + FString::FromInt(index) + FString(":") + FString::FromInt(sensorOut.lidarPointsOut.Num()) + FString("//");
		}
		if (sensorOut.depthArrayOut.Num() > 0) {
			bytesCount += sensorOut.depthArrayOut.Num() * 4;
			bytesInfo += FString("Depth") + FString::FromInt(index) + FString(":") + FString::FromInt(sensorOut.depthArrayOut.Num()) + FString("//");
		}
		if (sensorOut.colorArrayOut.Num() > 0) {
			bytesCount += sensorOut.colorArrayOut.Num() * 4;
			bytesInfo += FString("Color") + FString::FromInt(index) + FString(":") + FString::FromInt(sensorOut.colorArrayOut.Num()) + FString("//");
		}
		index++;
	}
	bytes.Init(0, bytesCount);
	uint32 offset = 0;
	for (const FLidarSensorOut& sensorOut : lidarSensorOuts) {
		if (sensorOut.lidarPointsOut.Num() > 0) {
			for (const FLidarPointCloudPoint& pp : sensorOut.lidarPointsOut) {
				memcpy(&bytes[offset], &(pp.Location), 4 * 3);
				offset += 12;
				memcpy(&bytes[offset], &(pp.Color.DWColor()), 4);
				offset += 4;
			}
		}
		if (sensorOut.depthArrayOut.Num() > 0) {
			memcpy(&bytes[offset], &sensorOut.depthArrayOut[0], sensorOut.depthArrayOut.Num() * 4);
			offset += sensorOut.depthArrayOut.Num() * 4;
		}
		if (sensorOut.colorArrayOut.Num() > 0) {
			memcpy(&bytes[offset], &sensorOut.colorArrayOut[0], sensorOut.colorArrayOut.Num() * 4);
			offset += sensorOut.colorArrayOut.Num() * 4;
		}
	}
}