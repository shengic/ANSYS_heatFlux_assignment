
'Private Type vector
'   x As Double
'   y As Double
'   z As Double
'End Type

'Private xV(), yV(), zV() As Double

Private xV() As Double
Private yV() As Double
Private zV() As Double

Public Sub initialized()
 
 'Dim xV() As Double
 'Dim yV() As Double
 'Dim zV() As Double
 
 xSource = form1.xSource.Value
 ySource = form1.ySource.Value
 zSource = form1.zSource.Value
 
 xTarget = form1.xTarget.Value
 yTarget = form1.yTarget.Value
 zTarget = form1.zTarget.Value
 
 xSide = form1.xSide.Value
 ySide = form1.ySide.Value
 zSide = form1.zSide.Value
 
' xsource = 0#
' ysource = 0#
' zsource = -8000#
 
' xtarget = 0#
' ytarget = 0#
' ztarget = 0#
 
' xside = 1#
' yside = 0#
' zside = zsource
 
 ReDim xV(2), yV(2), zV(2)
 
 zV(0) = xTarget - xSource
 zV(1) = yTarget - ySource
 zV(2) = zTarget - zSource
 
 xV(0) = xSide - xSource
 xV(1) = ySide - ySource
 xV(2) = zSide - zSource
 
' cross product to get y vector zV x xV
yV(0) = zV(1) * xV(2) - zV(2) * xV(1)
yV(1) = zV(2) * xV(0) - zV(0) * xV(2)
yV(2) = zV(0) * xV(1) - zV(1) * xV(0)

 xV = orthonormalize(xV)
 yV = orthonormalize(yV)
 zV = orthonormalize(zV)
 
form1.xV.Caption = "Base vector X: (" & xV(0) & ", " & xV(1) & ", " & xV(2) & ")"
form1.yV.Caption = "Base vector Y: (" & yV(0) & ", " & yV(1) & ", " & yV(2) & ")"
form1.zV.Caption = "Base vector Z: (" & zV(0) & ", " & zV(1) & ", " & zV(2) & ")"
 
End Sub

' not used
Public Sub mapAnsysCoordinateToMrad2(x As Double, y As Double, z As Double)
  Dim vx, vY, vz As Double
  Dim angleX As Double
  Dim angleY As Double

' source vector is pointing to the target

  vx = x - xSource
  vY = y - ySource
  vz = z - zSource
  
'  xmrad = WorksheetFunction.Atan2(z, x)
'  ymrad = WorksheetFunction.Atan2(z, y)
End Sub


Public Function orthonormalize(ByRef v() As Double) As Double()
  Dim magnitude As Double
  magnitude = Sqr(v(0) ^ 2 + v(1) ^ 2 + v(2) ^ 2)
  v(0) = v(0) / magnitude
  v(1) = v(1) / magnitude
  v(2) = v(2) / magnitude
  ReDim orthonormalize(2)
  orthonormalize = v
End Function

Public Function getVX() As Double()
  ReDim getVX(2)
   getVX = xV
End Function

Public Function getVY() As Double()
  ReDim getVY(2)
   getVY = yV
End Function

Public Function getVZ() As Double()
  ReDim getVY(2)
   getVZ = zV
End Function