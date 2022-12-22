


from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(120, 130)
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox = QtWidgets.QComboBox(Form)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout.addWidget(self.comboBox, 0, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "AI欢乐斗地主"))
        self.WinRate.setText(_translate("Form", "胜率：--%"))
        self.InitCard.setText(_translate("Form", "开始"))
        self.UserHandCards.setText(_translate("Form", "手牌"))
        self.LPlayedCard.setText(_translate("Form", "上家出牌区域"))
        self.RPlayedCard.setText(_translate("Form", "下家出牌区域"))
        self.PredictedCard.setText(_translate("Form", "AI出牌区域"))
        self.ThreeLandlordCards.setText(_translate("Form", "三张底牌"))
        self.Stop.setText(_translate("Form", "停止"))

if __name__ == '__main__':
    retranslateUi()