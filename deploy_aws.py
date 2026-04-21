"""
AWS S3 정적 호스팅 원클릭 배포 스크립트
사용법: python deploy_aws.py [--bucket 버킷명] [--region 리전]

사전 조건:
  pip install boto3
  aws configure  (액세스 키 설정)
"""
import sys, argparse, subprocess
from pathlib import Path

ROOT      = Path(__file__).parent
HTML      = ROOT / "app" / "dashboard_live.html"
BUCKET    = "alices-project-storage"
S3_KEY    = "pet-health-ai/dashboard/index.html"
REGION    = "ap-northeast-2"

def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",       default=BUCKET)
    parser.add_argument("--region",       default=REGION)
    parser.add_argument("--no-browser",   action="store_true")
    parser.add_argument("--skip-generate",action="store_true",
                        help="dashboard_live.html이 이미 있으면 재생성 건너뜀")
    args = parser.parse_args()

    try:
        import boto3
    except ImportError:
        print("❌ boto3 미설치 — pip install boto3")
        sys.exit(1)

    # ① 대시보드 HTML 생성
    if not args.skip_generate or not HTML.exists():
        print("📊 실데이터 HTML 생성 중...")
        env_flag = ["--no-browser"] if args.no_browser else []
        # run_dashboard.py 에 --no-browser 플래그 추가 필요 시 아래 직접 import
        import importlib.util, types
        spec = importlib.util.spec_from_file_location("run_dashboard", ROOT / "run_dashboard.py")
        # 간단히 subprocess로 실행
        cmd = [sys.executable, str(ROOT / "run_dashboard.py")]
        result = subprocess.run(cmd + (["--no-browser"] if args.no_browser else []),
                                capture_output=False)
        if result.returncode != 0:
            print("❌ HTML 생성 실패")
            sys.exit(1)
    else:
        print(f"⏭️  HTML 생성 건너뜀 (--skip-generate)")

    if not HTML.exists():
        print(f"❌ {HTML} 파일 없음")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=args.region)
    bucket = args.bucket

    # ② 버킷 생성 (이미 있으면 무시)
    print(f"🪣  S3 버킷 확인: {bucket}")
    try:
        s3.head_bucket(Bucket=bucket)
        print("   ✓ 버킷 이미 존재")
    except s3.exceptions.NoSuchBucket if hasattr(s3.exceptions,"NoSuchBucket") else Exception:
        try:
            if args.region == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": args.region},
                )
            print(f"   ✓ 버킷 생성 완료")
        except Exception as e:
            print(f"   ℹ️  버킷 생성 시도: {e}")

    # ③ Public Access Block 해제
    try:
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": False,
                "IgnorePublicAcls": False,
                "BlockPublicPolicy": False,
                "RestrictPublicBuckets": False,
            },
        )
        print("   ✓ 퍼블릭 액세스 허용")
    except Exception as e:
        print(f"   ⚠️  퍼블릭 액세스 설정: {e}")

    # ④ 버킷 정책 적용
    import json
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{bucket}/*",
        }],
    })
    try:
        s3.put_bucket_policy(Bucket=bucket, Policy=policy)
        print("   ✓ 퍼블릭 읽기 정책 적용")
    except Exception as e:
        print(f"   ⚠️  버킷 정책: {e}")

    # ⑤ 정적 웹사이트 설정
    try:
        s3.put_bucket_website(
            Bucket=bucket,
            WebsiteConfiguration={"IndexDocument": {"Suffix": "index.html"}},
        )
        print("   ✓ 정적 웹사이트 호스팅 활성화")
    except Exception as e:
        print(f"   ⚠️  웹사이트 설정: {e}")

    # ⑥ HTML 파일 업로드
    size_kb = HTML.stat().st_size // 1024
    print(f"📤 업로드: {S3_KEY} ({size_kb}KB)")
    s3.upload_file(
        str(HTML), bucket, S3_KEY,
        ExtraArgs={"ContentType": "text/html; charset=utf-8"},
    )

    url = f"https://{bucket}.s3.{args.region}.amazonaws.com/{S3_KEY}"
    print(f"\n✅ 배포 완료!")
    print(f"🌐 접속 URL: {url}")

    if not args.no_browser:
        import webbrowser
        webbrowser.open(url)

if __name__ == "__main__":
    main()
