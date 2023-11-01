import torch
import torchvision.models as models
import utils

# 모델 불러오기
rnet, _ = utils.get_model('R110_C10')

print(rnet)

input()
# 테스트 데이터
test_input = torch.randn(1, 3, 224, 224)  # 예: 배치 크기 1, 3채널, 224x224 이미지

# 중간 블록까지의 모델을 정의
class PartialResNet(torch.nn.Module):
    def __init__(self, original_model, cut_at_layer):
        super(PartialResNet, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:cut_at_layer])

    def forward(self, x):
        return self.features(x)

# 예를 들어, 첫 번째 블록 이후를 확인하려면
cut_at_layer = 5  # 이 값은 모델 구조를 바탕으로 설정
partial_model = PartialResNet(rnet, cut_at_layer)

# 중간 출력 확인
with torch.no_grad():
    test_output = partial_model(test_input)
    print(test_output.shape)
