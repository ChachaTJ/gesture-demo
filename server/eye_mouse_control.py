import os
import sys
import cv2
import pygame
import numpy as np
import pyautogui
import time

# EyeGestures 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
eye_gestures_path = os.path.join(current_dir, 'EyeGestures')
if eye_gestures_path not in sys.path:
    sys.path.append(eye_gestures_path)

# 디버깅용: sys.path 출력
# print(f"DEBUG: sys.path: {sys.path}")

try:
    # eyeGestures 패키지 내부 구조에 따라 import 경로 조정
    # EyeGestures/eyeGestures/__init__.py 가 있으므로 eyeGestures 패키지로 인식되어야 함
    from eyeGestures.utils import VideoCapture
    from eyeGestures import EyeGestures_v2
except ImportError as e:
    print(f"ImportError: {e}")
    # 혹시 모르니 EyeGestures 내부를 한번 더 추가
    sys.path.append(os.path.join(eye_gestures_path, 'eyeGestures'))
    try:
        from utils import VideoCapture
        from eyeGestures import EyeGestures_v2
    except ImportError as e2:
        print(f"Retry ImportError: {e2}")
        print("EyeGestures 라이브러리를 찾을 수 없습니다. 'EyeGestures' 폴더가 현재 디렉토리에 있는지 확인해주세요.")
        sys.exit(1)

# PyAutoGUI 설정
pyautogui.FAILSAFE = False  # 코너로 가면 멈추는 기능 해제 (필요시 True)
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

def main():
    pygame.init()
    pygame.font.init()

    # 전체 화면으로 설정하여 캘리브레이션 정확도 높임
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Eye Mouse Calibration & Control")
    
    font = pygame.font.Font(None, 48)
    
    # EyeGestures 초기화
    gestures = EyeGestures_v2()
    cap = VideoCapture(0)

    # 캘리브레이션 포인트 생성
    x = np.arange(0.1, 1.0, 0.4) # 0.1, 0.5, 0.9
    y = np.arange(0.1, 1.0, 0.4)
    xx, yy = np.meshgrid(x, y)
    calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
    n_points = len(calibration_map)
    np.random.shuffle(calibration_map)
    
    gestures.uploadCalibrationMap(calibration_map, context="mouse_control")
    gestures.setFixation(1.0) # Fixation threshold

    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x = 0
    prev_y = 0
    
    # 상태 변수
    is_calibrating = True

    print("캘리브레이션을 시작합니다. 화면의 파란 점을 응시해주세요.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: # q 누르면 종료
                    running = False
                elif event.key == pygame.K_ESCAPE:
                    running = False

        ret, frame = cap.read()
        if not ret:
            continue
            
        # 프레임 전처리
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1) # 거울 모드

        # EyeGestures Step
        # 캘리브레이션 중일 때는 calibrate=True
        calibrate_mode = (iterator < n_points)
        
        event_result, cevent = gestures.step(
            frame, 
            calibrate_mode, 
            SCREEN_WIDTH, 
            SCREEN_HEIGHT, 
            context="mouse_control"
        )

        if event_result is None:
            continue

        # 화면 그리기
        screen.fill((0, 0, 0))
        
        # 카메라 피드 작게 표시 (우측 하단)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(frame_surface, (SCREEN_WIDTH - 320, SCREEN_HEIGHT - 240))

        if calibrate_mode:
            # 캘리브레이션 진행 중
            if cevent and cevent.point is not None:
                # 포인트가 변경되었는지 확인
                if cevent.point[0] != prev_x or cevent.point[1] != prev_y:
                    iterator += 1
                    prev_x = cevent.point[0]
                    prev_y = cevent.point[1]
                
                # 캘리브레이션 점 그리기
                # cevent.point는 화면 좌표
                pygame.draw.circle(screen, (0, 0, 255), cevent.point, 20) # 파란 점
                
                # 진행 상황 텍스트
                text_surface = font.render(f"Calibration: {iterator}/{n_points}", True, (255, 255, 255))
                screen.blit(text_surface, (50, 50))
                
                instruction = font.render("Look at the BLUE circle", True, (255, 255, 255))
                screen.blit(instruction, (SCREEN_WIDTH//2 - 200, 100))

        else:
            # 캘리브레이션 완료 -> 마우스 제어 모드
            if is_calibrating:
                print("캘리브레이션 완료! 마우스 제어를 시작합니다.")
                is_calibrating = False
                # 전체화면 해제하고 작은 창으로 변경 (선택 사항이지만, 마우스 제어 확인을 위해)
                # pygame.display.set_mode((320, 240)) 
                # 하지만 전체화면 유지하고 투명하게 하거나, 그냥 오버레이처럼 쓰는게 나을수도 있음.
                # 여기서는 일단 전체화면 유지하되, 중앙에 안내 문구 표시
            
            # 시선 위치 표시 (빨간 점)
            gaze_point = event_result.point
            pygame.draw.circle(screen, (255, 0, 0), gaze_point, 15)
            
            # 마우스 이동
            # 좌표가 화면 밖으로 나가지 않도록 클램핑
            target_x = max(0, min(SCREEN_WIDTH, gaze_point[0]))
            target_y = max(0, min(SCREEN_HEIGHT, gaze_point[1]))
            
            # 부드러운 이동을 위해 pyautogui.moveTo의 duration 사용 가능하지만, 
            # 실시간성을 위해 바로 이동. 필요시 보간 로직 추가.
            pyautogui.moveTo(target_x, target_y)

            # 안내 문구
            status_text = font.render("Mouse Control Mode (Press 'Q' to quit)", True, (0, 255, 0))
            screen.blit(status_text, (50, 50))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
