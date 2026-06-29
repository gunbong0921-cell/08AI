# 모듈 입포트
from flask import Flask, request, render_template
import os
import joblib

# Flask 애플리케이션 설정
app = Flask(__name__)

# 학습된 모델 로드
pklfile = os.path.dirname(__file__) + "/lang/freq.pkl"
clf = joblib.load(pklfile)

# 텍스트 입력 양식 및 판정 결과 출력
@app.route('/', methods=['GET', 'POST'])
def index():
  text = request.form.get('text', '')
  msg = ''
  if text:
    lang = detect_lang(text)
    msg = "판정 결과: " + lang
  return render_template('08lang-skin.html', text=text, msg=msg)

def detect_lang(text):
  code_a, code_z = (ord("a"), ord("z"))
  cnt = [0 for i in range(26)]  
  for ch in text:
    n = ord(ch) - code_a
    if 0 <= n < 26: cnt[n] += 1
  total = sum(cnt)
  if total == 0: return "입력이 없습니다"
  freq = list(map(lambda n: n / total, cnt))
  res = clf.predict([freq])
  
  lang_dic = {"en": "영어", "fr": "프랑스어", "id": "인도네시아어", "tl": "타갈로그어"}
  return lang_dic.get(res[0], "알 수 없는 언어")

if __name__ == "__main__":
  app.run(debug=True)   
   