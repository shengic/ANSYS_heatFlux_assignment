'Const defaultX As Double = 100
'Const epsilon As Double = 0.05
Public inode As Long
Public xmrad As Double
Public ymrad As Double
Public powerDensity As Double

Public Sub initialized(i As Long, x As Double, y As Double, power As Double)
inode = i
xmrad = x
ymrad = y
powerDensity = power
End Sub