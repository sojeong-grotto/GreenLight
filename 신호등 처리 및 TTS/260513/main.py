import AI
import tts
import Operation
import subprocess
import multiprocessing as mp
import time

        

if __name__ == "__main__":

    tts_worker = tts.TTSWorker('voice_list.json')
    res_queue = mp.Queue(maxsize=5) # 큐 크기 제한 ==> 큐크기 조절해봐야함(딜레이 될 수 도 있음)
    res_queue = mp.Queue()
    # 첫번째 parameter는 처리할 이미지 데이터(임의로 문자열로 대체)
    AiProcess = mp.Process(target = AI.inference_process, args = ("이미지 데이터", res_queue))
    AiProcess.daemon = True # 메인이 죽으면 같이 죽도록 설정

    try:
        while   not (tts_worker._flag == True):
            time.sleep(0.1)
        
        AiProcess.start()
        while AiProcess.is_alive():
            if not res_queue.empty():
                Operation.operator(res_queue)
            else:
                time.sleep(0.01) # CPU 과부하 방지용 미세 대기

    # 결과 대기 스레드 — 메인 흐름 블로킹 방지 (임시 코드, 좀 더 확인 필요)
    
    except KeyboardInterrupt:
        print("\n[Main] 프로그램을 종료합니다...")
    finally:
        # 종료 처리를 아주 신중하게 진행
        print("[Main] 자식 프로세스 종료 중...")
        
        # 1. 프로세스 종료 신호
        AiProcess.terminate()
        
        # 2. 큐의 내부 버퍼 비우기 (데드락 방지 핵심)
        # 큐에 데이터가 남아있으면 join이 안 될 수 있음
        res_queue.cancel_join_thread() 
        
        # 3. 정리 대기 (최대 1~2초만 기다림)
        AiProcess.join(timeout=2)
        
        if AiProcess.is_alive():
            print("[Main] 프로세스가 정상적으로 종료되지 않아 강제 중단합니다.")
            
        print("[Main] 모든 정리가 완료되었습니다.")