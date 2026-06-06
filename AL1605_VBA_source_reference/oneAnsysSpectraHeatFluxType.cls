Public element As Integer
Public xmrad As Double
Public ymrad As Double
Public x, y, z As Double


' this module is to interpolate the power density based on the input xmrad and ymrad
' and the spectra element number

Public Sub initialized(ielement As Integer, x0 As Double, y0 As Double, z0 As Double)
 element = ielement
 x = x0
 y = y0
 z = z0
 Call transferXYZtoXYmrad
End Sub


Public Sub transferXYZtoXYmrad()

End Sub